// Function: sub_1312DB0
// Address: 0x1312db0
//
int __fastcall sub_1312DB0(__int64 a1, unsigned int a2)
{
  __int64 **v2; // rax
  __int64 *v3; // r13

  if ( pthread_mutex_trylock(&stru_4F96A80) )
  {
    sub_130AD90((__int64)&unk_4F96A40);
    byte_4F96AA8 = 1;
  }
  ++qword_4F96A78;
  if ( a1 != qword_4F96A70 )
  {
    ++qword_4F96A68;
    qword_4F96A70 = a1;
  }
  v2 = (__int64 **)(*(_QWORD *)&dword_5060A08 + 8LL * a2);
  v3 = *v2;
  if ( !*v2 || (*v2 = (__int64 *)1, v3 == (__int64 *)1) )
  {
    byte_4F96AA8 = 0;
    return pthread_mutex_unlock(&stru_4F96A80);
  }
  else
  {
    byte_4F96AA8 = 0;
    pthread_mutex_unlock(&stru_4F96A80);
    return sub_1311920(a1, v3);
  }
}
