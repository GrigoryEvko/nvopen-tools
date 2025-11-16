// Function: sub_27C3F00
// Address: 0x27c3f00
//
__int64 __fastcall sub_27C3F00(__int64 **a1, __int64 a2)
{
  __int64 *v3; // rcx
  __int64 v5; // rax
  __int64 v6; // rsi
  int v7; // eax
  int v8; // edi
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r8
  __int64 v12; // rdi
  unsigned int v13; // r12d
  unsigned __int64 v15; // r13
  __int64 v16; // r14
  __int64 v17; // rax
  _BYTE *v18; // rdi
  __int64 v19; // r14
  __int64 v20; // rsi
  _QWORD *v21; // rax
  _QWORD *v22; // rdx
  int v23; // eax
  int v24; // r9d

  v3 = *a1;
  v5 = **a1;
  v6 = *(_QWORD *)(v5 + 8);
  v7 = *(_DWORD *)(v5 + 24);
  if ( !v7 )
  {
LABEL_20:
    v12 = 0;
    goto LABEL_4;
  }
  v8 = v7 - 1;
  v9 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( a2 != *v10 )
  {
    v23 = 1;
    while ( v11 != -4096 )
    {
      v24 = v23 + 1;
      v9 = v8 & (v23 + v9);
      v10 = (__int64 *)(v6 + 16LL * v9);
      v11 = *v10;
      if ( a2 == *v10 )
        goto LABEL_3;
      v23 = v24;
    }
    goto LABEL_20;
  }
LABEL_3:
  v12 = v10[1];
LABEL_4:
  if ( *a1[1] == v12 )
  {
    v15 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v15 == a2 + 48 )
      goto LABEL_27;
    if ( !v15 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v15 - 24) - 30 > 0xA )
LABEL_27:
      BUG();
    if ( *(_BYTE *)(v15 - 24) == 31 )
    {
      v16 = v3[2];
      v17 = sub_D47930(v12);
      v13 = sub_B19720(v16, a2, v17);
      if ( (_BYTE)v13 )
      {
        v18 = *(_BYTE **)(v15 - 120);
        if ( *v18 != 17 )
          return 0;
        v19 = *a1[1];
        v20 = *(_QWORD *)(v15 - 32LL * sub_AC30F0((__int64)v18) - 56);
        if ( *(_BYTE *)(v19 + 84) )
        {
          v21 = *(_QWORD **)(v19 + 64);
          v22 = &v21[*(unsigned int *)(v19 + 76)];
          if ( v21 != v22 )
          {
            while ( v20 != *v21 )
            {
              if ( v22 == ++v21 )
                goto LABEL_17;
            }
            return 1;
          }
LABEL_17:
          sub_27C3750(**a1, *a1[1], (__int64)(*a1 + 7), (*a1)[1]);
          return v13;
        }
        if ( !sub_C8CA60(v19 + 56, v20) )
          goto LABEL_17;
      }
    }
  }
  return 1;
}
