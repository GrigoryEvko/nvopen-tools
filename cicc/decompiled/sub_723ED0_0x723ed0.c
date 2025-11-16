// Function: sub_723ED0
// Address: 0x723ed0
//
char *__fastcall sub_723ED0(__int64 a1, int a2)
{
  char *v2; // r12
  char *v3; // rax
  char *v5; // rax
  __time_t timer[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = 0;
  if ( !(unsigned int)sub_723E40(a1, timer) )
    return v2;
  v3 = ctime(timer);
  v2 = v3;
  if ( !a2 )
    return v2;
  v5 = sub_721BB0(v3, 10);
  if ( !v5 )
    return v2;
  *v5 = 0;
  return v2;
}
