// Function: sub_2CE0930
// Address: 0x2ce0930
//
__int64 __fastcall sub_2CE0930(_QWORD *a1, __int64 a2, __int64 a3, unsigned int *a4)
{
  __int64 result; // rax
  unsigned __int8 *v7; // rax
  unsigned __int64 v8; // rdi
  unsigned __int8 *v9; // rsi
  _QWORD *v10; // r10
  unsigned __int64 v11; // r11
  _QWORD *v12; // rax
  _QWORD *v13; // r9
  __int64 *v14; // r13
  __int64 *v15; // rbx
  __int64 v16; // r15
  unsigned __int8 v17; // al
  unsigned int v18; // edx
  unsigned __int8 v19; // [rsp+7h] [rbp-C9h]
  char *v21; // [rsp+10h] [rbp-C0h] BYREF
  char v22; // [rsp+20h] [rbp-B0h] BYREF
  void *v23; // [rsp+90h] [rbp-40h]

  if ( !a1[18] )
    return 0;
  if ( *(_DWORD *)(*(_QWORD *)(a3 + 8) + 8LL) >> 8 )
    return 0;
  v7 = sub_CF22E0((unsigned __int8 *)a3);
  v8 = a1[16];
  v9 = v7;
  v10 = *(_QWORD **)(a1[15] + 8 * ((unsigned __int64)v7 % v8));
  v11 = (unsigned __int64)v7 % v8;
  if ( !v10 )
    return 0;
  v12 = (_QWORD *)*v10;
  if ( v9 != *(unsigned __int8 **)(*v10 + 8LL) )
  {
    do
    {
      v13 = (_QWORD *)*v12;
      if ( !*v12 )
        return 0;
      v10 = v12;
      if ( v11 != v13[1] % v8 )
        return 0;
      v12 = (_QWORD *)*v12;
    }
    while ( v9 != (unsigned __int8 *)v13[1] );
  }
  v14 = (__int64 *)*v10;
  if ( !*v10 )
    return 0;
  v15 = (__int64 *)*v14;
  if ( !*v14 )
    goto LABEL_17;
  do
  {
    if ( v15[1] % v8 != v11 || v9 != (unsigned __int8 *)v15[1] )
    {
      if ( v14 != v15 )
        break;
      return 0;
    }
    v15 = (__int64 *)*v15;
  }
  while ( v15 );
LABEL_17:
  while ( 1 )
  {
    v16 = v14[2];
    v17 = (a2 == v16) | sub_B19DB0(a1[45], v16, a2);
    if ( v17 )
      break;
    v14 = (__int64 *)*v14;
    if ( v14 == v15 )
      return 0;
  }
  v19 = v17;
  sub_23D0AB0((__int64)&v21, a2, 0, 0, 0);
  v18 = *((_DWORD *)v14 + 6);
  if ( v18 > 6 )
  {
    v18 = (v18 == 101) + 15;
  }
  else if ( v18 )
  {
    switch ( v18 )
    {
      case 1u:
      case 4u:
        break;
      case 3u:
        v18 = 2;
        break;
      case 5u:
        v18 = 8;
        break;
      case 6u:
        v18 = 32;
        break;
      default:
        goto LABEL_24;
    }
  }
  else
  {
LABEL_24:
    v18 = 15;
  }
  *a4 = v18;
  nullsub_61();
  v23 = &unk_49DA100;
  nullsub_63();
  result = v19;
  if ( v21 != &v22 )
  {
    _libc_free((unsigned __int64)v21);
    return v19;
  }
  return result;
}
