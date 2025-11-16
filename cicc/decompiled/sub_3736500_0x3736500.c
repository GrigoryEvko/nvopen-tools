// Function: sub_3736500
// Address: 0x3736500
//
void __fastcall sub_3736500(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdx
  unsigned __int8 v6; // al
  __int64 v7; // rcx
  __int64 v8; // rdi
  const void *v9; // rax
  size_t v10; // rdx

  v5 = *(_QWORD *)(a2 + 8);
  v6 = *(_BYTE *)(v5 - 16);
  if ( (v6 & 2) != 0 )
    v7 = *(_QWORD *)(v5 - 32);
  else
    v7 = v5 - 16 - 8LL * ((v6 >> 2) & 0xF);
  v8 = *(_QWORD *)(v7 + 8);
  if ( v8 )
  {
    v9 = (const void *)sub_B91420(v8);
    if ( v10 )
      sub_324AD70(a1, a3, 3, v9, v10);
    v5 = *(_QWORD *)(a2 + 8);
  }
  sub_3249DD0(a1, a3, v5);
}
