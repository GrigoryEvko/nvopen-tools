// Function: sub_15BC830
// Address: 0x15bc830
//
__int64 __fastcall sub_15BC830(__int64 *a1, int a2, __int64 a3, __int64 a4, int a5, int a6, unsigned int a7, char a8)
{
  __int64 *v8; // r10
  __int16 v10; // r13
  __int64 v13; // r9
  __int64 result; // rax
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rdi
  int v19; // esi
  int v20; // eax
  unsigned int v21; // edi
  __int64 *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // [rsp+18h] [rbp-78h]
  int v25; // [rsp+20h] [rbp-70h]
  __int64 v26; // [rsp+28h] [rbp-68h]
  int v27; // [rsp+30h] [rbp-60h]
  __int64 v28; // [rsp+30h] [rbp-60h]
  int v29; // [rsp+38h] [rbp-58h]
  _BYTE v31[24]; // [rsp+40h] [rbp-50h] BYREF
  int v32; // [rsp+58h] [rbp-38h] BYREF
  int v33[13]; // [rsp+5Ch] [rbp-34h] BYREF

  v8 = a1;
  v10 = a2;
  if ( a7 )
  {
LABEL_4:
    v15 = *v8;
    *(_QWORD *)&v31[16] = a3;
    v27 = (int)v8;
    *(_OWORD *)v31 = 0;
    v16 = v15 + 720;
    v17 = sub_161E980(56, 3);
    v18 = v17;
    if ( v17 )
    {
      v19 = v27;
      v28 = v17;
      sub_1623D80(v17, v19, 11, a7, (unsigned int)v31, 3, 0, 0);
      v18 = v28;
      *(_WORD *)(v28 + 2) = v10;
      *(_QWORD *)(v28 + 24) = 0;
      *(_QWORD *)(v28 + 32) = a4;
      *(_DWORD *)(v28 + 48) = a5;
      *(_QWORD *)(v28 + 40) = 0;
      *(_DWORD *)(v28 + 52) = a6;
    }
    return sub_15BC690(v18, a7, v16);
  }
  v13 = *a1;
  *(_DWORD *)v31 = a2;
  *(_QWORD *)&v31[8] = a3;
  *(_QWORD *)&v31[16] = a4;
  v32 = a5;
  v33[0] = a6;
  v24 = v13;
  v25 = *(_DWORD *)(v13 + 744);
  v26 = *(_QWORD *)(v13 + 728);
  if ( !v25 )
    goto LABEL_3;
  v20 = sub_15B4F20((int *)v31, (__int64 *)&v31[8], (__int64 *)&v31[16], &v32, v33);
  v8 = a1;
  v21 = (v25 - 1) & v20;
  v22 = (__int64 *)(v26 + 8LL * v21);
  v23 = *v22;
  if ( *v22 == -8 )
    goto LABEL_3;
  v29 = 1;
  while ( v23 == -16
       || *(_DWORD *)v31 != *(unsigned __int16 *)(v23 + 2)
       || *(_OWORD *)&v31[8] != __PAIR128__(
                                  *(_QWORD *)(v23 + 32),
                                  *(_QWORD *)(v23 + 8 * (2LL - *(unsigned int *)(v23 + 8))))
       || v32 != *(_DWORD *)(v23 + 48)
       || v33[0] != *(_DWORD *)(v23 + 52) )
  {
    v21 = (v25 - 1) & (v29 + v21);
    v22 = (__int64 *)(v26 + 8LL * v21);
    v23 = *v22;
    if ( *v22 == -8 )
      goto LABEL_3;
    ++v29;
  }
  if ( v22 == (__int64 *)(*(_QWORD *)(v24 + 728) + 8LL * *(unsigned int *)(v24 + 744)) || (result = *v22) == 0 )
  {
LABEL_3:
    result = 0;
    if ( !a8 )
      return result;
    goto LABEL_4;
  }
  return result;
}
