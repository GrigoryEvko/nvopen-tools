// Function: sub_289A440
// Address: 0x289a440
//
__int64 __fastcall sub_289A440(__int64 a1, _BYTE *a2, _BYTE *a3, __int64 a4, unsigned int **a5)
{
  __int64 v7; // rax
  unsigned int v8; // r15d
  int v9; // eax
  __int64 v10; // r8
  _BYTE *v12; // [rsp+8h] [rbp-68h] BYREF
  _QWORD v13[4]; // [rsp+10h] [rbp-60h] BYREF
  char v14; // [rsp+30h] [rbp-40h]
  char v15; // [rsp+31h] [rbp-3Fh]

  v15 = 1;
  v13[0] = "vec.start";
  v14 = 3;
  v7 = sub_A81850(a5, a2, a3, (__int64)v13, 0, 0);
  v12 = (_BYTE *)v7;
  if ( *(_BYTE *)v7 != 17 )
    goto LABEL_4;
  v8 = *(_DWORD *)(v7 + 32);
  if ( v8 <= 0x40 )
  {
    v10 = a1;
    if ( !*(_QWORD *)(v7 + 24) )
      return v10;
    goto LABEL_4;
  }
  v9 = sub_C444A0(v7 + 24);
  v10 = a1;
  if ( v8 != v9 )
  {
LABEL_4:
    v15 = 1;
    v13[0] = "vec.gep";
    v14 = 3;
    return sub_921130(a5, a4, a1, &v12, 1, (__int64)v13, 0);
  }
  return v10;
}
