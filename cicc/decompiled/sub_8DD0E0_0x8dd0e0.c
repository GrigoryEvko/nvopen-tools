// Function: sub_8DD0E0
// Address: 0x8dd0e0
//
__int64 __fastcall sub_8DD0E0(__int64 a1, _DWORD *a2, int *a3, _DWORD *a4, _DWORD *a5)
{
  int v7; // edx
  _BOOL4 v8; // eax
  __int64 result; // rax
  int v10; // edx

  if ( !dword_4D04440 )
  {
    *a3 = 0;
    *a2 = 0;
    *a5 = 0;
    dword_4F6058C = 0;
    dword_4F60590 = 0;
    if ( dword_4F077BC )
      dword_4F60588 = qword_4F077A8 > 0x9E33u;
    else
      dword_4F60588 = 0;
    goto LABEL_10;
  }
  dword_4F6058C = 0;
  dword_4F60590 = 0;
  v7 = dword_4D03B68;
  *a3 = 0;
  v8 = 0;
  *a2 = 0;
  *a5 = 0;
  if ( dword_4F077BC )
    v8 = qword_4F077A8 > 0x9E33u;
  dword_4F60588 = v8;
  if ( !v7 )
  {
LABEL_10:
    result = sub_8D9600(a1, sub_8D1CF0, 0x57u);
    v10 = dword_4F6058C;
    if ( dword_4F6058C )
      dword_4D03B68 = 1;
    if ( !dword_4D04440 )
    {
      *a2 = dword_4F60590;
      *a3 = v10;
      if ( (_DWORD)result )
        goto LABEL_7;
    }
    goto LABEL_6;
  }
  if ( !dword_4D04440 )
  {
    *a3 = 0;
    result = 0;
    if ( !unk_4F072F4 )
      goto LABEL_7;
    goto LABEL_17;
  }
LABEL_6:
  result = 0;
  if ( !unk_4F072F4 )
  {
LABEL_7:
    *a4 = 0;
    return result;
  }
LABEL_17:
  result = sub_8DD010(a1);
  *a4 = result;
  return result;
}
