// Function: sub_278B590
// Address: 0x278b590
//
__int64 __fastcall sub_278B590(__int64 a1, __int64 **a2, __int64 a3)
{
  unsigned __int8 *v3; // rax
  size_t v4; // rdx
  __int64 v5; // rbx
  __int64 *v6; // r15
  unsigned __int64 v7; // rdx
  int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // r12
  char v14; // al
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  int v17; // edx
  __int64 v18; // rax
  bool v19; // cf
  __int64 v20; // rdx
  __int64 v21[8]; // [rsp+10h] [rbp-C0h] BYREF
  unsigned __int64 v22[2]; // [rsp+50h] [rbp-80h] BYREF
  _BYTE v23[112]; // [rsp+60h] [rbp-70h] BYREF

  if ( *((_DWORD *)a2 + 2) == 1 && (sub_B196A0(*(_QWORD *)(a3 + 24), **a2, *(_QWORD *)(a1 + 40)), v14) )
  {
    v15 = **a2;
    v16 = *(_QWORD *)(v15 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v16 == v15 + 48 )
    {
      v20 = 0;
    }
    else
    {
      if ( !v16 )
LABEL_27:
        BUG();
      v17 = *(unsigned __int8 *)(v16 - 24);
      v18 = v16 - 24;
      v19 = (unsigned int)(v17 - 30) < 0xB;
      v20 = 0;
      if ( v19 )
        v20 = v18;
    }
    return sub_278B200((_BYTE **)*a2 + 1, a1, v20);
  }
  else
  {
    v22[0] = (unsigned __int64)v23;
    v22[1] = 0x800000000LL;
    sub_11D2BF0((__int64)v21, (__int64)v22);
    v3 = (unsigned __int8 *)sub_BD5D20(a1);
    sub_11D2C80(v21, *(_QWORD *)(a1 + 8), v3, v4);
    v5 = (__int64)&(*a2)[5 * *((unsigned int *)a2 + 2)];
    if ( (__int64 *)v5 != *a2 )
    {
      v6 = *a2;
      do
      {
        if ( *((_DWORD *)v6 + 4) != 3 )
        {
          v11 = *v6;
          if ( !(unsigned __int8)sub_11D3030(v21, *v6)
            && (v11 != *(_QWORD *)(a1 + 40) || *((_DWORD *)v6 + 4) > 1u || a1 != v6[1]) )
          {
            v7 = *(_QWORD *)(*v6 + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v7 == *v6 + 48 )
            {
              v9 = 0;
            }
            else
            {
              if ( !v7 )
                goto LABEL_27;
              v8 = *(unsigned __int8 *)(v7 - 24);
              v9 = v7 - 24;
              if ( (unsigned int)(v8 - 30) >= 0xB )
                v9 = 0;
            }
            v10 = sub_278B200((_BYTE **)v6 + 1, a1, v9);
            sub_11D33F0(v21, v11, v10);
          }
        }
        v6 += 5;
      }
      while ( (__int64 *)v5 != v6 );
    }
    v12 = sub_11D7E40(v21, *(_QWORD *)(a1 + 40));
    sub_11D2C20(v21);
    if ( (_BYTE *)v22[0] != v23 )
      _libc_free(v22[0]);
    return v12;
  }
}
