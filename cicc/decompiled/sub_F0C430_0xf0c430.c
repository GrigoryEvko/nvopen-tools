// Function: sub_F0C430
// Address: 0xf0c430
//
__int64 __fastcall sub_F0C430(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v5; // [rsp+10h] [rbp-20h]

  v3 = *(_QWORD *)(a2 - 32);
  if ( v3 )
  {
    if ( *(_BYTE *)v3 )
    {
      v3 = 0;
    }
    else if ( *(_QWORD *)(v3 + 24) != *(_QWORD *)(a2 + 80) )
    {
      v3 = 0;
    }
  }
  if ( (unsigned __int8)sub_B2DD60(v3) )
    return sub_DF9D20(*(_QWORD *)(a1 + 8));
  else
    return v5;
}
