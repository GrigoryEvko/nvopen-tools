// Function: sub_3571720
// Address: 0x3571720
//
__int64 __fastcall sub_3571720(__int64 *a1, __int64 a2, char a3)
{
  char v3; // r8
  __int64 v4; // r9
  unsigned int v5; // r12d
  __int64 v6; // r10
  unsigned int v8; // r12d
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r11
  __int64 v13; // rax
  unsigned int v14; // eax
  __int64 *v15; // r14
  __int64 *v16; // rbx
  unsigned __int64 i; // rdx
  __int64 v18; // rdi
  unsigned int v19; // ecx
  __int64 v20; // rsi
  __int64 *v21; // rbx
  __int64 *v22; // r13
  __int64 v23; // rsi
  __int64 v24; // rdi
  int v25; // eax
  int v26; // ebx
  __int64 v27[4]; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v28; // [rsp+30h] [rbp-B0h]
  __int64 v29; // [rsp+38h] [rbp-A8h]
  unsigned int v30; // [rsp+40h] [rbp-A0h]
  __int64 v31; // [rsp+48h] [rbp-98h]
  __int64 v32; // [rsp+50h] [rbp-90h]
  __int64 *v33; // [rsp+58h] [rbp-88h]
  __int64 v34; // [rsp+60h] [rbp-80h]
  _BYTE v35[32]; // [rsp+68h] [rbp-78h] BYREF
  __int64 *v36; // [rsp+88h] [rbp-58h]
  __int64 v37; // [rsp+90h] [rbp-50h]
  _QWORD v38[9]; // [rsp+98h] [rbp-48h] BYREF

  v3 = a3;
  v4 = *a1;
  v5 = *(_DWORD *)(*a1 + 24);
  v6 = *(_QWORD *)(*a1 + 8);
  if ( v5 )
  {
    v8 = v5 - 1;
    v9 = v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
    {
LABEL_3:
      v5 = *((_DWORD *)v10 + 2);
      v3 = a3 | (v5 != 0);
    }
    else
    {
      v25 = 1;
      while ( v11 != -4096 )
      {
        v26 = v25 + 1;
        v9 = v8 & (v25 + v9);
        v10 = (__int64 *)(v6 + 16LL * v9);
        v11 = *v10;
        if ( a2 == *v10 )
          goto LABEL_3;
        v25 = v26;
      }
      v5 = 0;
    }
  }
  if ( !v3 )
  {
    v13 = a1[3];
    v27[0] = (__int64)a1;
    v27[1] = v4;
    v27[2] = v13;
    v33 = (__int64 *)v35;
    v27[3] = 0;
    v28 = 0;
    v29 = 0;
    v30 = 0;
    v31 = 0;
    v32 = 0;
    v34 = 0x400000000LL;
    v36 = v38;
    v37 = 0;
    v38[0] = 0;
    v38[1] = 1;
    v14 = sub_3570F10(v27, a2);
    v15 = v33;
    v5 = v14;
    v16 = &v33[(unsigned int)v34];
    if ( v33 != v16 )
    {
      for ( i = (unsigned __int64)v33; ; i = (unsigned __int64)v33 )
      {
        v18 = *v15;
        v19 = (unsigned int)((__int64)((__int64)v15 - i) >> 3) >> 7;
        v20 = 4096LL << v19;
        if ( v19 >= 0x1E )
          v20 = 0x40000000000LL;
        ++v15;
        sub_C7D6A0(v18, v20, 16);
        if ( v16 == v15 )
          break;
      }
    }
    v21 = v36;
    v22 = &v36[2 * (unsigned int)v37];
    if ( v36 != v22 )
    {
      do
      {
        v23 = v21[1];
        v24 = *v21;
        v21 += 2;
        sub_C7D6A0(v24, v23, 16);
      }
      while ( v22 != v21 );
      v22 = v36;
    }
    if ( v22 != v38 )
      _libc_free((unsigned __int64)v22);
    if ( v33 != (__int64 *)v35 )
      _libc_free((unsigned __int64)v33);
    sub_C7D6A0(v28, 16LL * v30, 8);
  }
  return v5;
}
