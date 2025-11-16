// Function: sub_A738C0
// Address: 0xa738c0
//
unsigned __int64 __fastcall sub_A738C0(__int64 *a1)
{
  __int64 v1; // rdi
  __int64 v3; // [rsp-8h] [rbp-8h]

  v1 = *a1;
  if ( v1 )
    return sub_A73860(v1);
  *((_BYTE *)&v3 - 4) = 0;
  return *(&v3 - 2);
}
