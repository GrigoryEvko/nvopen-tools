// Function: sub_F3F2F0
// Address: 0xf3f2f0
//
__int64 __fastcall sub_F3F2F0(__int64 a1, __int64 a2)
{
  int v2; // ebx
  int v4; // ebx

  v2 = sub_F3E970(a1, a2);
  if ( !sub_AA5B70(a1) || !(unsigned __int8)sub_AEA460(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 40LL)) )
    return v2 | (unsigned int)sub_F3B6A0(a1, a2);
  v4 = sub_F3D570(a1) | v2;
  return v4 | (unsigned int)sub_F3B6A0(a1, a2);
}
