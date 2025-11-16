// Function: sub_816350
// Address: 0x816350
//
__int64 sub_816350()
{
  __int64 v0; // rbx
  __int64 i; // rbx

  v0 = qword_4F07288;
  sub_816050(*(_QWORD *)(qword_4F07288 + 104));
  for ( i = *(_QWORD *)(v0 + 168); i; i = *(_QWORD *)(i + 112) )
  {
    if ( (*(_BYTE *)(i + 124) & 1) == 0 )
      sub_8160E0(*(_QWORD *)(i + 128));
  }
  return sub_76C540(qword_4F07288, (__int64 (__fastcall *)(_QWORD))sub_816050);
}
