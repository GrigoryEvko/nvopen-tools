// Function: sub_2221060
// Address: 0x2221060
//
void __fastcall sub_2221060(_QWORD *a1)
{
  _QWORD *v2; // rax
  volatile signed __int32 *v3; // rdi
  signed __int32 v4; // eax

  *a1 = off_4A05DD0;
  v2 = (_QWORD *)a1[4];
  v3 = (volatile signed __int32 *)a1[3];
  v2[3] = 0;
  v2[7] = 0;
  v2[9] = 0;
  v2[11] = 0;
  if ( &_pthread_key_create )
  {
    v4 = _InterlockedExchangeAdd(v3 + 2, 0xFFFFFFFF);
  }
  else
  {
    v4 = *((_DWORD *)v3 + 2);
    *((_DWORD *)v3 + 2) = v4 - 1;
  }
  if ( v4 == 1 )
    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v3 + 8LL))(v3);
  sub_220C140(a1);
}
