// Function: sub_B58E70
// Address: 0xb58e70
//
__int64 __fastcall sub_B58E70(char **a1, unsigned int a2)
{
  char *v2; // rdx
  char v3; // al
  __int64 v4; // r8

  v2 = *a1;
  v3 = **a1;
  if ( v3 == 4 )
    return *(_QWORD *)(*(_QWORD *)(*((_QWORD *)v2 + 17) + 8LL * a2) + 136LL);
  v4 = 0;
  if ( (unsigned __int8)(v3 - 5) <= 0x1Fu )
    return v4;
  return *((_QWORD *)v2 + 17);
}
