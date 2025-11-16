// Function: sub_1312C70
// Address: 0x1312c70
//
__int64 __fastcall sub_1312C70(__int64 a1, __int64 a2, int *a3)
{
  unsigned __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rcx
  unsigned int v7; // r12d
  int v9; // esi
  int v10; // eax

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
  if ( (*(_QWORD *)&dword_5060A08 || (*(_QWORD *)&dword_5060A08 = sub_131C440(a1, a2, 32752, 64)) != 0)
    && (qword_4F96AB0 || (unsigned int)dword_4F96AB8 <= 0xFFD)
    && (v4 = sub_1311F90(a1)) != 0 )
  {
    v5 = qword_4F96AB0;
    if ( qword_4F96AB0 )
    {
      v6 = *(_QWORD *)qword_4F96AB0;
      *(_QWORD *)qword_4F96AB0 = v4;
      v7 = 0;
      qword_4F96AB0 = v6;
      *a3 = (v5 - *(_QWORD *)&dword_5060A08) >> 3;
    }
    else
    {
      v9 = dword_4F96AB8;
      v7 = 0;
      v10 = dword_4F96AB8;
      *(_QWORD *)(*(_QWORD *)&dword_5060A08 + 8LL * (unsigned int)dword_4F96AB8) = v4;
      *a3 = v9;
      dword_4F96AB8 = v10 + 1;
    }
  }
  else
  {
    v7 = 1;
  }
  byte_4F96AA8 = 0;
  pthread_mutex_unlock(&stru_4F96A80);
  return v7;
}
