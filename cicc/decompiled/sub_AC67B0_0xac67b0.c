// Function: sub_AC67B0
// Address: 0xac67b0
//
__int64 __fastcall sub_AC67B0(__int64 a1, _QWORD *a2, __int64 **a3)
{
  unsigned int v4; // r13d
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // r14
  int v12; // eax
  __int64 v13; // r8
  __int64 *v14; // r9
  __int64 i; // rcx
  __int64 *v16; // rdx
  __int64 v17; // rsi
  char v18; // al
  __int64 *v19; // rdx
  unsigned int v20; // ecx
  unsigned int v21; // eax
  _QWORD *j; // rbx
  char v23; // al
  __int64 *v24; // [rsp+8h] [rbp-A8h]
  unsigned int v25; // [rsp+8h] [rbp-A8h]
  unsigned int v26; // [rsp+8h] [rbp-A8h]
  int v27; // [rsp+10h] [rbp-A0h]
  __int64 *v28; // [rsp+10h] [rbp-A0h]
  __int64 *v29; // [rsp+10h] [rbp-A0h]
  __int64 *v30; // [rsp+18h] [rbp-98h]
  unsigned int v31; // [rsp+18h] [rbp-98h]
  unsigned int v32; // [rsp+18h] [rbp-98h]
  unsigned int v33; // [rsp+20h] [rbp-90h]
  __int64 *v34; // [rsp+20h] [rbp-90h]
  __int64 *v35; // [rsp+20h] [rbp-90h]
  int v36; // [rsp+28h] [rbp-88h]
  __int64 v37; // [rsp+30h] [rbp-80h]
  __int64 v38; // [rsp+38h] [rbp-78h]
  _QWORD v39[4]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v40; // [rsp+60h] [rbp-50h] BYREF
  _QWORD *v41; // [rsp+68h] [rbp-48h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v38 = *(_QWORD *)(a1 + 8);
    v37 = sub_C33690();
    v10 = sub_C33340(a1, a2, v7, v8, v9);
    v11 = v10;
    if ( v37 == v10 )
    {
      sub_C3C5A0(v39, v10, 1);
      sub_C3C5A0(&v40, v11, 2);
    }
    else
    {
      sub_C36740(v39, v37, 1);
      sub_C36740(&v40, v37, 2);
    }
    v12 = sub_C42050(a2);
    v13 = v4 - 1;
    v14 = 0;
    v36 = 1;
    for ( i = (unsigned int)v13 & v12; ; i = v33 & v20 )
    {
      v16 = (__int64 *)(v38 + 32LL * (unsigned int)i);
      v17 = *v16;
      if ( *a2 == *v16 )
      {
        v25 = i;
        v28 = v14;
        v31 = v13;
        v34 = (__int64 *)(v38 + 32LL * (unsigned int)i);
        if ( v11 == *a2 )
        {
          v21 = sub_C3E590(a2);
          i = v25;
          v14 = v28;
          v13 = v31;
          v16 = v34;
        }
        else
        {
          v21 = sub_C33D00(a2);
          v16 = v34;
          v13 = v31;
          v14 = v28;
          i = v25;
        }
        v4 = v21;
        if ( (_BYTE)v21 )
        {
          *a3 = v16;
          goto LABEL_16;
        }
        v17 = *v16;
      }
      if ( v39[0] == v17 )
      {
        v26 = i;
        v29 = v14;
        v32 = v13;
        v35 = v16;
        if ( v11 == v17 )
        {
          v23 = sub_C3E590(v16);
          i = v26;
          v14 = v29;
          v13 = v32;
          v16 = v35;
        }
        else
        {
          v23 = sub_C33D00(v16);
          v16 = v35;
          v13 = v32;
          v14 = v29;
          i = v26;
        }
        if ( v23 )
          break;
      }
      v27 = i;
      v30 = v14;
      v33 = v13;
      v24 = v16;
      v18 = sub_AC2B80(v16, &v40, (__int64)v16, i, v13);
      v13 = v33;
      if ( v30 || (v19 = v24, !v18) )
        v19 = v30;
      v14 = v19;
      v20 = v36 + v27;
      ++v36;
    }
    if ( !v14 )
      v14 = v16;
    v4 = 0;
    *a3 = v14;
LABEL_16:
    if ( v11 == v40 )
    {
      if ( v41 )
      {
        for ( j = &v41[3 * *(v41 - 1)]; v41 != j; sub_91D830(j) )
          j -= 3;
        j_j_j___libc_free_0_0(j - 1);
      }
    }
    else
    {
      sub_C338F0(&v40);
    }
    sub_91D830(v39);
  }
  else
  {
    *a3 = 0;
  }
  return v4;
}
