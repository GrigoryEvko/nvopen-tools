// Function: sub_CE7F70
// Address: 0xce7f70
//
__int64 __fastcall sub_CE7F70(_BYTE *a1, __int64 a2)
{
  __int64 v2; // rbp
  int v4; // [rsp-Ch] [rbp-Ch] BYREF
  __int64 v5; // [rsp-8h] [rbp-8h]

  if ( *a1 > 3u )
    return 0;
  v5 = v2;
  return sub_CE7ED0((__int64)a1, *(const void **)a2, *(_QWORD *)(a2 + 8), &v4);
}
