// Function: sub_749E60
// Address: 0x749e60
//
__int64 __fastcall sub_749E60(__int64 a1, _DWORD *a2, __int64 (__fastcall **a3)(const char *, _QWORD))
{
  unsigned __int8 v5; // r14
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // rdi
  __int64 result; // rax
  __int64 i; // rcx
  unsigned __int64 v11; // rax
  _BYTE v12[96]; // [rsp+0h] [rbp-60h] BYREF

  v5 = *(_BYTE *)(a1 + 177);
  if ( *a2 )
    (*a3)(" ", a3);
  if ( v5 == 2 )
  {
    (*a3)("__attribute((neon_vector_type(", a3);
  }
  else
  {
    if ( v5 > 2u )
    {
      if ( v5 != 3 )
        sub_721090();
      (*a3)("__attribute((neon_polyvector_type(", a3);
      v8 = *(_QWORD *)(a1 + 168);
      if ( v8 )
        goto LABEL_8;
LABEL_12:
      for ( i = *(_QWORD *)(a1 + 160); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      v11 = *(_QWORD *)(a1 + 128) / *(_QWORD *)(i + 128);
      if ( v11 > 9 )
      {
LABEL_15:
        sub_622470(v11, v12);
LABEL_20:
        (*a3)(v12, a3);
        goto LABEL_9;
      }
LABEL_19:
      v12[1] = 0;
      v12[0] = v11 + 48;
      goto LABEL_20;
    }
    if ( !v5 )
    {
      (*a3)("__attribute((vector_size(", a3);
      v8 = *(_QWORD *)(a1 + 168);
      if ( v8 )
        goto LABEL_8;
      v11 = *(_QWORD *)(a1 + 128);
      if ( v11 > 9 )
        goto LABEL_15;
      goto LABEL_19;
    }
    (*a3)("__attribute((ext_vector_type(", a3);
  }
  v8 = *(_QWORD *)(a1 + 168);
  if ( !v8 )
    goto LABEL_12;
LABEL_8:
  sub_748000(v8, 0, (__int64)a3, v6, v7);
LABEL_9:
  result = (*a3)(")))", a3);
  *a2 = 1;
  return result;
}
