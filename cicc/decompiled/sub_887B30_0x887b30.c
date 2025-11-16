// Function: sub_887B30
// Address: 0x887b30
//
_DWORD *__fastcall sub_887B30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rbx
  _DWORD *result; // rax
  _DWORD *v8; // rcx
  _DWORD *v9; // rdx

  qword_4F60030 = 0;
  dword_4F5FFBC = 0;
  qword_4D049B8 = 0;
  dword_4F5FFB8 = 0;
  qword_4D049A8 = 0;
  dword_4F5FFB4 = 0;
  qword_4D049A0 = 0;
  dword_4F5FFB0 = 0;
  qword_4D049B0 = 0;
  dword_4F600B0 = 0;
  unk_4D049D8 = 0;
  unk_4D049D0 = 0;
  unk_4D049C8 = 0;
  unk_4D049C0 = 0;
  qword_4D04998 = 0;
  unk_4D04990 = 0;
  *(_QWORD *)&dword_4D04988 = 0;
  qword_4D04980 = 0;
  qword_4F600B8 = 0;
  unk_4D04978 = 0;
  qword_4F5FFC8 = 0;
  qword_4D04970 = 0;
  unk_4D04968 = 0;
  unk_4D04A48 = 0;
  unk_4D04A40 = 0;
  dword_4F066AC = 1;
  if ( dword_4D03FE8[0] )
    *((_QWORD *)qword_4F066A0 + dword_4F066A8) = qword_4D03FF0;
  else
    dword_4F066A8 = sub_880E90(a1, a2, 0, a4, a5, a6);
  if ( !dword_4D03FE8[0] )
    dword_4F04C3C = 1;
  v6 = (_QWORD *)sub_823970(16);
  result = &qword_4F5FED8;
  qword_4F5FED8 = v6;
  if ( v6 )
  {
    result = (_DWORD *)sub_823970(0x4000);
    v8 = result;
    v9 = result + 4096;
    do
    {
      if ( result )
      {
        *result = 0;
        result[1] = 0;
      }
      result += 4;
    }
    while ( result != v9 );
    *v6 = v8;
    v6[1] = 1023;
  }
  return result;
}
