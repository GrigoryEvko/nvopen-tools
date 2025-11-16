// Function: sub_1312E80
// Address: 0x1312e80
//
int __fastcall sub_1312E80(__int64 a1, unsigned int a2)
{
  __int64 v2; // rdx
  __int64 **v3; // rax
  __int64 *v4; // r13

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
  v2 = qword_4F96AB0;
  v3 = (__int64 **)(*(_QWORD *)&dword_5060A08 + 8LL * a2);
  v4 = *v3;
  qword_4F96AB0 = (__int64)v3;
  *v3 = (__int64 *)v2;
  byte_4F96AA8 = 0;
  if ( (unsigned __int64)v4 <= 1 )
    return pthread_mutex_unlock(&stru_4F96A80);
  pthread_mutex_unlock(&stru_4F96A80);
  return sub_1311920(a1, v4);
}
