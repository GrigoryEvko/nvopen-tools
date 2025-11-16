// Function: sub_8DF7D0
// Address: 0x8df7d0
//
__int64 __fastcall sub_8DF7D0(__int64 a1, __int64 a2, _DWORD *a3)
{
  __int64 v5; // rdi
  unsigned int v6; // r14d
  int v8; // r15d
  __int64 v9; // r12
  __int64 v10; // rdi
  int v11[13]; // [rsp+Ch] [rbp-34h] BYREF

  if ( a3 )
    *a3 = 0;
  v5 = a2;
  v6 = sub_8D2F30(a2, a1);
  if ( v6 )
    goto LABEL_10;
  if ( sub_8D3D10(a2) && sub_8D3D10(a1) )
  {
    v9 = sub_8D4870(a2);
    v10 = sub_8D4870(a1);
    return (unsigned int)sub_8DF240(v10, v9, v11, 1, 1, a3, 0, 0) == 0;
  }
  if ( (unsigned int)sub_8D3230(a2, a1) )
  {
    v8 = sub_8D3110(a2);
    if ( v8 == (unsigned int)sub_8D3110(a1) )
    {
      v5 = a2;
LABEL_10:
      v9 = sub_8D46C0(v5);
      v10 = sub_8D46C0(a1);
      return (unsigned int)sub_8DF240(v10, v9, v11, 1, 1, a3, 0, 0) == 0;
    }
  }
  return v6;
}
