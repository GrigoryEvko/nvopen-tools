// Function: sub_947E80
// Address: 0x947e80
//
__int64 __fastcall sub_947E80(__int64 a1, __int64 a2, __int64 a3, int a4, char a5)
{
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  _QWORD v13[3]; // [rsp+0h] [rbp-50h] BYREF
  int v14; // [rsp+18h] [rbp-38h]
  char v15; // [rsp+1Ch] [rbp-34h]

  if ( !sub_91B770(*(_QWORD *)a2) )
    sub_91B8A0("expected expression with aggregate type!", (_DWORD *)(a2 + 36), 1);
  v13[0] = a1;
  v13[1] = a1 + 48;
  v13[2] = a3;
  v14 = a4;
  v15 = a5;
  return sub_947AB0(v13, a2, v8, v9, v10, v11);
}
