// Function: sub_320CF50
// Address: 0x320cf50
//
void __fastcall sub_320CF50(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 *v5; // r14
  __int64 v6; // rax
  unsigned __int8 v7; // dl
  unsigned __int8 **v8; // rax
  unsigned __int8 *v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  unsigned int v15; // esi
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // r8
  int v20; // r11d
  __int64 v21; // rdx
  __int64 v22; // r9
  __int64 *v23; // rdi
  __int64 v24; // r10
  __int64 v25; // rdi
  int v26; // ecx
  __int64 v27; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v28; // [rsp+18h] [rbp-28h] BYREF

  v5 = *(__int64 **)(a3 + 16);
  v27 = a3;
  if ( !v5 )
  {
    v15 = *(_DWORD *)(a1 + 864);
    v16 = a1 + 840;
    if ( v15 )
    {
      v17 = a3;
      v18 = *(_QWORD *)(a1 + 848);
      v19 = v15 - 1;
      v20 = 1;
      v21 = (unsigned int)v19 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v22 = (unsigned int)v21;
      v23 = (__int64 *)(v18 + 112LL * (unsigned int)v21);
      v24 = *v23;
      if ( v17 == *v23 )
      {
LABEL_8:
        v25 = (__int64)(v23 + 1);
LABEL_9:
        sub_31FE5D0(v25, a2, v21, v18, v19, v22);
        return;
      }
      while ( v24 != -4096 )
      {
        if ( !v5 && v24 == -8192 )
          v5 = v23;
        v22 = (unsigned int)(v20 + 1);
        v21 = (unsigned int)v19 & (v20 + (_DWORD)v21);
        v23 = (__int64 *)(v18 + 112LL * (unsigned int)v21);
        v24 = *v23;
        if ( v17 == *v23 )
          goto LABEL_8;
        ++v20;
      }
      v26 = *(_DWORD *)(a1 + 856);
      if ( !v5 )
        v5 = v23;
      ++*(_QWORD *)(a1 + 840);
      v18 = (unsigned int)(v26 + 1);
      v28 = v5;
      if ( 4 * (int)v18 < 3 * v15 )
      {
        v21 = v15 - *(_DWORD *)(a1 + 860) - (unsigned int)v18;
        if ( (unsigned int)v21 > v15 >> 3 )
          goto LABEL_13;
        goto LABEL_12;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 840);
      v28 = 0;
    }
    v15 *= 2;
LABEL_12:
    sub_320BE40(v16, v15);
    sub_31FB250(v16, &v27, &v28);
    v17 = v27;
    v5 = v28;
    v18 = (unsigned int)(*(_DWORD *)(a1 + 856) + 1);
LABEL_13:
    *(_DWORD *)(a1 + 856) = v18;
    if ( *v5 != -4096 )
      --*(_DWORD *)(a1 + 860);
    *v5 = v17;
    v25 = (__int64)(v5 + 1);
    v5[1] = (__int64)(v5 + 3);
    v5[2] = 0x100000000LL;
    goto LABEL_9;
  }
  v6 = *a2;
  v7 = *(_BYTE *)(*a2 - 16);
  if ( (v7 & 2) != 0 )
    v8 = *(unsigned __int8 ***)(v6 - 32);
  else
    v8 = (unsigned __int8 **)(v6 - 16 - 8LL * ((v7 >> 2) & 0xF));
  v9 = sub_AF34D0(*v8);
  v10 = sub_320C1A0(a1, (__int64)v5, (__int64)v9);
  sub_31FE5D0(v10, a2, v11, v12, v13, v14);
}
