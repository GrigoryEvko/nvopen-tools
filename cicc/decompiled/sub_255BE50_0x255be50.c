// Function: sub_255BE50
// Address: 0x255be50
//
__int64 __fastcall sub_255BE50(__int64 a1, const void ***a2)
{
  __int64 (*v2)(void); // rax
  char v3; // r14
  __int64 (__fastcall *v4)(__int64); // rax
  char v5; // al
  __int64 (__fastcall *v6)(__int64); // rax
  char v7; // al
  __int64 (__fastcall *v8)(__int64); // rax
  char v9; // al
  __int64 v10; // r13
  __int64 v11; // rbx
  const void **v12; // r12
  __int64 v13; // r13

  v2 = *(__int64 (**)(void))(*(_QWORD *)a1 + 16LL);
  if ( (char *)v2 == (char *)sub_2505E40 )
    v3 = *(_BYTE *)(a1 + 17);
  else
    v3 = v2();
  v4 = (__int64 (__fastcall *)(__int64))(*a2)[2];
  if ( v4 == sub_2505E40 )
    v5 = *((_BYTE *)a2 + 17);
  else
    v5 = v4((__int64)a2);
  if ( v3 != v5 )
    return 0;
  v6 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL);
  if ( v6 == sub_2505E40 )
    v7 = *(_BYTE *)(a1 + 17);
  else
    v7 = v6(a1);
  if ( v7
    || ((v8 = (__int64 (__fastcall *)(__int64))(*a2)[2], v8 != sub_2505E40)
      ? (v9 = v8((__int64)a2))
      : (v9 = *((_BYTE *)a2 + 17)),
        v9) )
  {
    if ( *(_BYTE *)(a1 + 200) == *((_BYTE *)a2 + 200) )
    {
      v10 = *(unsigned int *)(a1 + 64);
      if ( v10 == *((_DWORD *)a2 + 16) )
      {
        v11 = *(_QWORD *)(a1 + 56);
        v12 = a2[7];
        v13 = v11 + 16 * v10;
        if ( v11 != v13 )
        {
          while ( 1 )
          {
            if ( *(_DWORD *)(v11 + 8) > 0x40u )
            {
              if ( !sub_C43C50(v11, v12) )
                return 0;
            }
            else if ( *(const void **)v11 != *v12 )
            {
              return 0;
            }
            v11 += 16;
            v12 += 2;
            if ( v13 == v11 )
              return 1;
          }
        }
        return 1;
      }
    }
    return 0;
  }
  return 1;
}
