// Function: sub_22163D0
// Address: 0x22163d0
//
const wchar_t *__fastcall sub_22163D0(const wchar_t **a1)
{
  const wchar_t *result; // rax

  result = *a1;
  if ( *a1 - 6 != (const wchar_t *)&unk_4FD67E0 )
  {
    if ( *(result - 2) > 0 )
      sub_22161B0(a1, 0, 0, 0);
    result = *a1;
    *((_DWORD *)*a1 - 2) = -1;
  }
  return result;
}
