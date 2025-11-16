// Function: sub_642070
// Address: 0x642070
//
__int64 __fastcall sub_642070(_DWORD *a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  __int64 v4; // rcx
  unsigned int v6; // r15d
  __int16 v7; // r14
  unsigned __int64 v8; // rax
  int v9; // [rsp+Ch] [rbp-104h] BYREF
  _BYTE v10[173]; // [rsp+10h] [rbp-100h] BYREF
  char v11; // [rsp+BDh] [rbp-53h]

  v3 = 3779;
  v4 = (unsigned int)*a1;
  if ( (int)v4 < 0 )
  {
    sub_7B8B50(a1, a2, a3, v4);
    if ( !(unsigned int)sub_7BE280(27, 125, 0, 0) )
      return 125;
    v6 = dword_4F063F8;
    v7 = unk_4F063FC;
    sub_6BA680(v10);
    if ( v11 == 1 )
    {
      if ( (int)sub_6210B0((__int64)v10, 0) <= 0 )
      {
        v3 = 3783;
        goto LABEL_8;
      }
      v8 = sub_620FD0((__int64)v10, &v9);
      if ( v9 || v8 > 0x7FFFFFFF )
      {
        v3 = 3782;
        goto LABEL_8;
      }
      *a1 = v8;
      v3 = 0;
    }
    else
    {
      if ( v11 != 12 )
      {
        v3 = v11 == 0 ? 3781 : 3784;
LABEL_8:
        *(_DWORD *)a2 = v6;
        *(_WORD *)(a2 + 4) = v7;
        goto LABEL_9;
      }
      if ( dword_4F04C44 == -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0 )
      {
        v3 = 3784;
        goto LABEL_8;
      }
      v3 = 0;
    }
LABEL_9:
    if ( (unsigned int)sub_7BE280(28, 18, 0, 0) )
      return v3;
    return 125;
  }
  return v3;
}
