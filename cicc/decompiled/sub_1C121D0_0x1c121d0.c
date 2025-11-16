// Function: sub_1C121D0
// Address: 0x1c121d0
//
__int64 __fastcall sub_1C121D0(__int64 a1, __int64 a2, char a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v10; // rdi
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  __int64 result; // rax
  __int64 v14; // rcx
  __int64 v15; // r15
  _QWORD *v16; // rbx
  _QWORD *v17; // r12
  int v18; // [rsp+2Ch] [rbp-144h] BYREF
  __int64 v19; // [rsp+30h] [rbp-140h] BYREF
  __int64 v20; // [rsp+38h] [rbp-138h]
  __int64 v21; // [rsp+40h] [rbp-130h]
  __int64 v22; // [rsp+48h] [rbp-128h]
  __int64 v23; // [rsp+50h] [rbp-120h] BYREF
  __int64 v24; // [rsp+58h] [rbp-118h]
  __int64 v25; // [rsp+60h] [rbp-110h]
  __int64 v26; // [rsp+68h] [rbp-108h]
  __int64 v27; // [rsp+70h] [rbp-100h] BYREF
  _QWORD *v28; // [rsp+78h] [rbp-F8h]
  __int64 v29; // [rsp+80h] [rbp-F0h]
  unsigned int v30; // [rsp+88h] [rbp-E8h]
  __int64 v31; // [rsp+90h] [rbp-E0h] BYREF
  _BYTE *v32; // [rsp+98h] [rbp-D8h]
  _BYTE *v33; // [rsp+A0h] [rbp-D0h]
  __int64 v34; // [rsp+A8h] [rbp-C8h]
  int v35; // [rsp+B0h] [rbp-C0h]
  _BYTE v36[184]; // [rsp+B8h] [rbp-B8h] BYREF

  *(_BYTE *)a1 = a3;
  *(_QWORD *)(a1 + 8) = a4;
  *(_QWORD *)(a1 + 16) = a2;
  *(_QWORD *)(a1 + 24) = a6;
  sub_1C08020(a4, a2, a5);
  if ( a3 )
  {
    result = sub_1C2F070(a2);
    if ( !(_BYTE)result )
      return result;
  }
  v10 = *(_QWORD *)(a2 + 80);
  v11 = 0;
  v12 = v10;
  if ( v10 == a2 + 72 )
  {
    v19 = 0;
    v20 = 0;
    v21 = 0;
    v22 = 0;
LABEL_6:
    v10 -= 24;
    goto LABEL_7;
  }
  do
  {
    v12 = *(_QWORD *)(v12 + 8);
    ++v11;
  }
  while ( a2 + 72 != v12 );
  result = (unsigned int)dword_4FBA260;
  if ( (unsigned int)dword_4FBA260 < v11 )
    return result;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  if ( v10 )
    goto LABEL_6;
LABEL_7:
  sub_1C0B8A0(v10, (__int64)&v19);
  v31 = 0;
  v32 = v36;
  v33 = v36;
  v34 = 16;
  v35 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v18 = 0;
  sub_1C0B430(a1, a2, (__int64)&v31, (__int64)&v19, (__int64)&v23, &v18);
  v14 = *(_QWORD *)(a1 + 8);
  v27 = 0;
  v15 = *(_QWORD *)(v14 + 104);
  v28 = 0;
  v29 = 0;
  v30 = 0;
  sub_1C0E0D0(v15, a2, a6, a7, (__int64)&v19, (__int64)&v31, (__int64)&v27);
  sub_1C10C40(a1, a2, v15, (__int64)&v23, &v18, (__int64)&v27);
  if ( v30 )
  {
    v16 = v28;
    v17 = &v28[5 * v30];
    do
    {
      if ( *v16 != -16 && *v16 != -8 )
        j___libc_free_0(v16[2]);
      v16 += 5;
    }
    while ( v17 != v16 );
  }
  j___libc_free_0(v28);
  j___libc_free_0(v24);
  if ( v33 != v32 )
    _libc_free((unsigned __int64)v33);
  return j___libc_free_0(v20);
}
