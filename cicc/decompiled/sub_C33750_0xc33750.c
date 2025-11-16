// Function: sub_C33750
// Address: 0xc33750
//
char __fastcall sub_C33750(__int64 a1)
{
  char *v1; // rax
  char result; // al

  v1 = (char *)sub_C94E20(qword_4F86290);
  if ( v1 )
    result = *v1;
  else
    result = qword_4F86290[2];
  if ( result )
    return *(_DWORD *)(a1 + 12) <= 0x20u;
  return result;
}
