// Function: sub_324CDB0
// Address: 0x324cdb0
//
char __fastcall sub_324CDB0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // r8
  __int64 v9; // rdx
  unsigned __int8 v10; // al
  __int64 *v11; // r14
  const void *v12; // rcx
  size_t v13; // rdx
  size_t v14; // r8

  v4 = a3 - 16;
  v5 = sub_324C6D0(a1, 47, a2, 0);
  LOBYTE(v6) = *(_BYTE *)(a3 - 16);
  if ( (v6 & 2) != 0 )
  {
    v7 = *(__int64 **)(a3 - 32);
    v8 = v7[1];
    if ( !v8 )
      goto LABEL_5;
  }
  else
  {
    v6 = 8LL * (((unsigned __int8)v6 >> 2) & 0xF);
    v7 = (__int64 *)(v4 - v6);
    v8 = *(_QWORD *)(v4 - v6 + 8);
    if ( !v8 )
      goto LABEL_5;
  }
  sub_32495E0(a1, v5, v8, 73);
  LOBYTE(v6) = *(_BYTE *)(a3 - 16);
  if ( (v6 & 2) != 0 )
  {
    v7 = *(__int64 **)(a3 - 32);
  }
  else
  {
    v6 = 8LL * (((unsigned __int8)v6 >> 2) & 0xF);
    v7 = (__int64 *)(v4 - v6);
  }
LABEL_5:
  if ( *v7 )
  {
    LOBYTE(v6) = sub_B91420(*v7);
    if ( v9 )
    {
      v10 = *(_BYTE *)(a3 - 16);
      if ( (v10 & 2) != 0 )
        v11 = *(__int64 **)(a3 - 32);
      else
        v11 = (__int64 *)(v4 - 8LL * ((v10 >> 2) & 0xF));
      v12 = (const void *)*v11;
      if ( *v11 )
      {
        v12 = (const void *)sub_B91420(*v11);
        v14 = v13;
      }
      else
      {
        v14 = 0;
      }
      LOBYTE(v6) = sub_324AD70(a1, v5, 3, v12, v14);
    }
  }
  if ( *(char *)(a3 + 1) < 0 )
  {
    LOBYTE(v6) = sub_3248C10((__int64)a1, 5u);
    if ( (_BYTE)v6 )
      LOBYTE(v6) = sub_3249FA0(a1, v5, 30);
  }
  return v6;
}
