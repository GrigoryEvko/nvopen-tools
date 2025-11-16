// Function: sub_7DF680
// Address: 0x7df680
//
_QWORD *__fastcall sub_7DF680(__int64 a1)
{
  _QWORD *result; // rax
  int v2; // esi
  __int64 v3; // rdx
  _QWORD *v4; // rcx
  _QWORD *v5; // rdx
  __int64 v6; // rcx
  _QWORD *v7; // rdx

  result = *(_QWORD **)(a1 + 232);
  if ( !result )
    return result;
  v2 = dword_4F04C64;
  if ( dword_4F04C64 >= 0 )
  {
    v3 = *(_QWORD *)(qword_4F04C68[0] + 24LL);
    if ( !v3 )
      v3 = qword_4F04C68[0] + 32LL;
    v4 = *(_QWORD **)(v3 + 96);
    if ( v4 )
      goto LABEL_6;
    v6 = unk_4F07288;
LABEL_18:
    *(_QWORD *)(v6 + 232) = result;
    goto LABEL_7;
  }
  v6 = unk_4F07288;
  v7 = *(_QWORD **)(unk_4F07288 + 232LL);
  if ( !v7 )
    goto LABEL_18;
  do
  {
    v4 = v7;
    v7 = (_QWORD *)*v7;
  }
  while ( v7 );
LABEL_6:
  *v4 = result;
  v2 = dword_4F04C64;
LABEL_7:
  if ( v2 >= 0 )
  {
    do
    {
      v5 = result;
      result = (_QWORD *)*result;
    }
    while ( result );
    result = *(_QWORD **)(qword_4F04C68[0] + 24LL);
    if ( !result )
      result = (_QWORD *)(qword_4F04C68[0] + 32LL);
    result[12] = v5;
  }
  *(_QWORD *)(a1 + 232) = 0;
  return result;
}
