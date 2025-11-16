// Function: sub_1EB5F30
// Address: 0x1eb5f30
//
__int64 __fastcall sub_1EB5F30(_QWORD *a1, int a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  unsigned int v6; // eax
  __int64 v7; // r15
  __int64 v8; // r8
  __int64 v11; // r13
  unsigned __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r12
  _BYTE *v15; // rsi
  _BYTE *v16; // rax
  __int64 v17; // rsi
  __int64 v18; // r8
  __int64 result; // rax
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 *v22; // rdi
  unsigned int v23; // eax
  __int64 v24; // rcx
  __int64 v25; // r12
  __int64 v26; // rdi
  _QWORD *v27; // rsi
  _QWORD *v28; // rdx
  __int64 *v29; // rcx
  unsigned int v30; // [rsp+Ch] [rbp-44h]
  __int64 v31[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a2 & 0x7FFFFFFF;
  v7 = a2 & 0x7FFFFFFF;
  v8 = 8 * v7;
  v11 = a1[33];
  v12 = *(unsigned int *)(v11 + 408);
  if ( (a2 & 0x7FFFFFFFu) >= (unsigned int)v12
    || (v13 = *(_QWORD *)(v11 + 400), (v14 = *(_QWORD *)(v13 + 8LL * v6)) == 0) )
  {
    v23 = v6 + 1;
    if ( (unsigned int)v12 < v23 )
    {
      v25 = v23;
      if ( v23 < v12 )
      {
        *(_DWORD *)(v11 + 408) = v23;
      }
      else if ( v23 > v12 )
      {
        if ( v23 > (unsigned __int64)*(unsigned int *)(v11 + 412) )
        {
          v30 = v23;
          sub_16CD150(v11 + 400, (const void *)(v11 + 416), v23, 8, 8 * a2, a6);
          v12 = *(unsigned int *)(v11 + 408);
          v8 = 8LL * (a2 & 0x7FFFFFFF);
          v23 = v30;
        }
        v24 = *(_QWORD *)(v11 + 400);
        v26 = *(_QWORD *)(v11 + 416);
        v27 = (_QWORD *)(v24 + 8 * v25);
        v28 = (_QWORD *)(v24 + 8 * v12);
        if ( v27 != v28 )
        {
          do
            *v28++ = v26;
          while ( v27 != v28 );
          v24 = *(_QWORD *)(v11 + 400);
        }
        *(_DWORD *)(v11 + 408) = v23;
        goto LABEL_15;
      }
    }
    v24 = *(_QWORD *)(v11 + 400);
LABEL_15:
    *(_QWORD *)(v24 + v8) = sub_1DBA290(a2);
    v14 = *(_QWORD *)(*(_QWORD *)(v11 + 400) + 8 * v7);
    sub_1DBB110((_QWORD *)v11, v14);
  }
  sub_21031A0(a1[34], v14, v12, v13, v8);
  v31[0] = v14;
  v15 = (_BYTE *)a1[88];
  if ( v15 == (_BYTE *)a1[89] )
  {
    sub_1EB5DA0((__int64)(a1 + 87), v15, v31);
    v16 = (_BYTE *)a1[88];
  }
  else
  {
    if ( v15 )
    {
      *(_QWORD *)v15 = v14;
      v15 = (_BYTE *)a1[88];
    }
    v16 = v15 + 8;
    a1[88] = v15 + 8;
  }
  v17 = a1[87];
  v18 = *((_QWORD *)v16 - 1);
  result = (__int64)&v16[-v17];
  v20 = (result >> 3) - 1;
  v21 = ((result >> 3) - 2) / 2;
  if ( v20 > 0 )
  {
    while ( 1 )
    {
      v22 = (__int64 *)(v17 + 8 * v21);
      v29 = (__int64 *)(v17 + 8 * v20);
      result = *v22;
      if ( *(float *)(v18 + 116) <= *(float *)(*v22 + 116) )
        break;
      *v29 = result;
      v20 = v21;
      result = (v21 - 1) / 2;
      if ( v21 <= 0 )
      {
        v29 = (__int64 *)(v17 + 8 * v21);
        break;
      }
      v21 = (v21 - 1) / 2;
    }
  }
  else
  {
    v29 = (__int64 *)(v17 + result - 8);
  }
  *v29 = v18;
  return result;
}
