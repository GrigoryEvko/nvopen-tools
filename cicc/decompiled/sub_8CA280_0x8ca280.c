// Function: sub_8CA280
// Address: 0x8ca280
//
__int64 __fastcall sub_8CA280(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx
  __int64 v3; // rdx

  result = a1;
  if ( qword_4F074B0 == qword_4F60258 && a1 && *qword_4D03FD0 )
  {
    if ( unk_4D03FC4 || unk_4D03FC0 && (*(_BYTE *)(a1 + 89) & 4) != 0 )
    {
      v2 = *(_QWORD *)(a1 + 32);
      if ( v2 )
        return *(_QWORD *)v2;
      sub_8C9400(a1, 11);
      result = a1;
      v3 = *(_QWORD *)(a1 + 32);
      if ( v3 )
        return *(_QWORD *)v3;
    }
    else
    {
      v2 = *(_QWORD *)(a1 + 32);
      if ( v2 )
        return *(_QWORD *)v2;
    }
  }
  return result;
}
