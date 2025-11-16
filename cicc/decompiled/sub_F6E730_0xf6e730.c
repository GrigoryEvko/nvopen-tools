// Function: sub_F6E730
// Address: 0xf6e730
//
__int64 __fastcall sub_F6E730(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rcx
  __int64 v11; // rax
  unsigned int v12; // ebx
  char v13; // dl
  char v14; // r15
  char v15; // r14
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rax
  int v20; // r13d
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 result; // rax
  __int64 v29; // rax
  char v30; // dl
  char v31; // r14
  unsigned __int64 v32; // rbx
  int v33; // r13d
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  unsigned __int64 v37; // rax
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  unsigned __int64 v41; // rbx
  int v42; // edx
  unsigned __int64 v43; // rax
  char v44; // r8
  char v45; // [rsp+Fh] [rbp-51h]

  LOWORD(v6) = sub_D4A1D0(a1, "llvm.loop.vectorize.enable", 0x1Au, a4, a5, a6);
  v10 = v6;
  LOWORD(v10) = BYTE1(v6);
  if ( BYTE1(v6) )
  {
    if ( !(_BYTE)v6 )
      return 6;
    v29 = sub_F6E040(a1, (__int64)"llvm.loop.vectorize.enable", v7, v10, v8, v9);
    v31 = v30;
    v32 = v29;
    v33 = v29;
    v37 = sub_D4A2B0(a1, "llvm.loop.interleave.count", 0x1Au, v34, v35, v36);
    v41 = HIDWORD(v32);
    v42 = v37;
    v43 = HIDWORD(v37);
    if ( v31 )
    {
      if ( (_BYTE)v41 != 1 && v33 == 1 && v42 == 1 && (_BYTE)v43 )
        return 6;
    }
    v44 = sub_D4A290(a1, "llvm.loop.isvectorized", 0x16u, v38, v39, v40);
    result = 5;
    if ( v44 )
      return 2;
    return result;
  }
  v11 = sub_F6E040(a1, (__int64)"llvm.loop.vectorize.enable", v7, v10, v8, v9);
  v12 = v11;
  v14 = v13;
  v15 = BYTE4(v11);
  v19 = sub_D4A2B0(a1, "llvm.loop.interleave.count", 0x1Au, v16, v17, v18);
  v20 = v19;
  v45 = BYTE4(v19);
  if ( (unsigned __int8)sub_D4A290(a1, "llvm.loop.isvectorized", 0x16u, v21, v22, v23) )
    return 2;
  if ( v14 )
  {
    if ( v15 != 1 )
    {
      if ( v12 == 1 )
      {
        if ( v20 == 1 && v45 )
          return 2;
        goto LABEL_4;
      }
      if ( !v15 )
      {
        if ( v12 > 1 )
          return 1;
        goto LABEL_4;
      }
    }
    result = 1;
    if ( v12 )
      return result;
  }
LABEL_4:
  if ( v20 <= 1 || (result = 1, !v45) )
  {
    if ( !(unsigned __int8)sub_F6E590(a1, (__int64)"llvm.loop.isvectorized", v24, v25, v26, v27) )
      return 0;
    return 2;
  }
  return result;
}
