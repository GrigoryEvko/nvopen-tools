// Function: sub_7B3E40
// Address: 0x7b3e40
//
__int64 sub_7B3E40()
{
  int v0; // edx
  unsigned __int8 *v1; // r14
  int v3[9]; // [rsp+Ch] [rbp-24h] BYREF

  v0 = 1;
  v1 = qword_4F06460;
  if ( dword_4F055C0[(char)*qword_4F06460 + 128] )
    return 0;
  do
  {
    if ( !(unsigned int)sub_7B3CF0(v1, v3, v0) )
      break;
    v1 += v3[0];
    v0 = dword_4F055C0[(char)*v1 + 128];
  }
  while ( !v0 );
  if ( qword_4F06460 == v1 )
    return 0;
  qword_4F06460 = v1;
  return 1;
}
