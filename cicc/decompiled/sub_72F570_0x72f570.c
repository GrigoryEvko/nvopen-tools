// Function: sub_72F570
// Address: 0x72f570
//
_BOOL8 __fastcall sub_72F570(__int64 a1)
{
  __int64 v1; // rbp
  char v3; // [rsp-Ch] [rbp-Ch] BYREF
  __int64 v4; // [rsp-8h] [rbp-8h]

  if ( *(_BYTE *)(a1 + 174) != 1 || !dword_4D04474 )
    return 0;
  v4 = v1;
  return (unsigned int)sub_72F500(a1, *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL), &v3, 1, 1)
      && (unsigned int)sub_72F530(a1) != 0;
}
