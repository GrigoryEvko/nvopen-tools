// Function: sub_CFDBA0
// Address: 0xcfdba0
//
__int64 __fastcall sub_CFDBA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  unsigned __int8 *v8; // rsi
  __int64 result; // rax
  _BYTE *v10; // r14
  _BYTE *v11; // rbx
  __int64 *v12; // rcx
  __int64 v13; // r8
  __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rdx
  signed __int64 v18; // rax
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // r9
  __int64 v21; // rdx
  char *v22; // r9
  __int64 v23; // rax
  __int64 v24; // rdx
  _BYTE *v25; // rbx
  __int64 v26; // r8
  __int64 *v27; // [rsp+0h] [rbp-280h]
  char *v28; // [rsp+8h] [rbp-278h]
  __int64 v29; // [rsp+8h] [rbp-278h]
  __int64 *v30; // [rsp+10h] [rbp-270h]
  __int64 v31; // [rsp+10h] [rbp-270h]
  __int64 *v32; // [rsp+10h] [rbp-270h]
  __int64 *v33; // [rsp+10h] [rbp-270h]
  _QWORD v34[2]; // [rsp+20h] [rbp-260h] BYREF
  __int64 v35; // [rsp+30h] [rbp-250h]
  int v36; // [rsp+38h] [rbp-248h]
  _BYTE *v37; // [rsp+40h] [rbp-240h] BYREF
  __int64 v38; // [rsp+48h] [rbp-238h]
  _BYTE v39[560]; // [rsp+50h] [rbp-230h] BYREF

  v8 = *(unsigned __int8 **)(a1 + 8);
  v37 = v39;
  v38 = 0x1000000000LL;
  result = (__int64)sub_CFC6F0(a2, v8, (__int64)&v37, a4, a5, a6);
  v10 = v37;
  v11 = &v37[32 * (unsigned int)v38];
  if ( v11 == v37 )
    goto LABEL_32;
  do
  {
    while ( 1 )
    {
      v12 = sub_CFD720(a1, *((_QWORD *)v10 + 2));
      result = *v12;
      v14 = 32LL * *((unsigned int *)v12 + 2);
      v8 = (unsigned __int8 *)(*v12 + v14);
      v15 = v14 >> 5;
      v16 = v14 >> 7;
      if ( v16 )
      {
        v17 = result + (v16 << 7);
        while ( *(_QWORD *)(result + 16) != a2 || *(_DWORD *)(result + 24) != *((_DWORD *)v10 + 6) )
        {
          if ( *(_QWORD *)(result + 48) == a2 && *(_DWORD *)(result + 56) == *((_DWORD *)v10 + 6) )
          {
            result += 32;
            goto LABEL_36;
          }
          if ( *(_QWORD *)(result + 80) == a2 && *(_DWORD *)(result + 88) == *((_DWORD *)v10 + 6) )
          {
            result += 64;
            goto LABEL_36;
          }
          if ( *(_QWORD *)(result + 112) == a2 && *(_DWORD *)(result + 120) == *((_DWORD *)v10 + 6) )
          {
            result += 96;
            goto LABEL_36;
          }
          result += 128;
          if ( v17 == result )
          {
            v15 = (__int64)&v8[-result] >> 5;
            goto LABEL_10;
          }
        }
        goto LABEL_36;
      }
LABEL_10:
      if ( v15 == 2 )
        goto LABEL_47;
      if ( v15 != 3 )
      {
        if ( v15 != 1 )
          goto LABEL_13;
LABEL_49:
        if ( *(_QWORD *)(result + 16) != a2 || *(_DWORD *)(result + 24) != *((_DWORD *)v10 + 6) )
          goto LABEL_13;
        goto LABEL_36;
      }
      if ( *(_QWORD *)(result + 16) != a2 || *(_DWORD *)(result + 24) != *((_DWORD *)v10 + 6) )
      {
        result += 32;
LABEL_47:
        if ( *(_QWORD *)(result + 16) != a2 || *(_DWORD *)(result + 24) != *((_DWORD *)v10 + 6) )
        {
          result += 32;
          goto LABEL_49;
        }
      }
LABEL_36:
      if ( v8 != (unsigned __int8 *)result )
        break;
LABEL_13:
      v35 = a2;
      v34[0] = 4;
      v34[1] = 0;
      if ( a2 != -4096 && a2 != 0 && a2 != -8192 )
      {
        v30 = v12;
        sub_BD73F0((__int64)v34);
        v12 = v30;
      }
      v36 = *((_DWORD *)v10 + 6);
      v18 = *((unsigned int *)v12 + 2);
      v19 = *((unsigned int *)v12 + 3);
      v20 = v18 + 1;
      v8 = (unsigned __int8 *)v18;
      if ( v18 + 1 > v19 )
      {
        if ( *v12 > (unsigned __int64)v34 || (unsigned __int64)v34 >= *v12 + 32 * v18 )
        {
          v33 = v12;
          sub_CFC2E0((__int64)v12, v20, v19, (__int64)v12, v13, v20);
          v12 = v33;
          v22 = (char *)v34;
          v18 = *((unsigned int *)v33 + 2);
          v21 = *v33;
          v8 = (unsigned __int8 *)v18;
        }
        else
        {
          v29 = *v12;
          v32 = v12;
          sub_CFC2E0((__int64)v12, v20, v19, (__int64)v12, v13, v20);
          v12 = v32;
          v21 = *v32;
          v18 = *((unsigned int *)v32 + 2);
          v22 = (char *)v34 + *v32 - v29;
          v8 = (unsigned __int8 *)v18;
        }
      }
      else
      {
        v21 = *v12;
        v22 = (char *)v34;
      }
      v23 = v21 + 32 * v18;
      if ( v23 )
      {
        *(_QWORD *)v23 = 4;
        v24 = *((_QWORD *)v22 + 2);
        *(_QWORD *)(v23 + 8) = 0;
        *(_QWORD *)(v23 + 16) = v24;
        if ( v24 != 0 && v24 != -4096 && v24 != -8192 )
        {
          v27 = v12;
          v28 = v22;
          v31 = v23;
          sub_BD6050((unsigned __int64 *)v23, *(_QWORD *)v22 & 0xFFFFFFFFFFFFFFF8LL);
          v12 = v27;
          v22 = v28;
          v23 = v31;
        }
        *(_DWORD *)(v23 + 24) = *((_DWORD *)v22 + 6);
        v8 = (unsigned __int8 *)*((unsigned int *)v12 + 2);
      }
      *((_DWORD *)v12 + 2) = (_DWORD)v8 + 1;
      result = v35;
      if ( v35 == -4096 || v35 == 0 || v35 == -8192 )
        break;
      v10 += 32;
      result = sub_BD60C0(v34);
      if ( v10 == v11 )
        goto LABEL_26;
    }
    v10 += 32;
  }
  while ( v10 != v11 );
LABEL_26:
  v25 = v37;
  v26 = 32LL * (unsigned int)v38;
  v10 = &v37[v26];
  if ( v37 != &v37[v26] )
  {
    do
    {
      result = *((_QWORD *)v10 - 2);
      v10 -= 32;
      if ( result != -4096 && result != 0 && result != -8192 )
        result = sub_BD60C0(v10);
    }
    while ( v25 != v10 );
    v10 = v37;
  }
LABEL_32:
  if ( v10 != v39 )
    return _libc_free(v10, v8);
  return result;
}
