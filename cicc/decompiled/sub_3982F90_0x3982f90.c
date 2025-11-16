// Function: sub_3982F90
// Address: 0x3982f90
//
void __fastcall sub_3982F90(__int64 **a1, __int64 a2, __int16 a3)
{
  _BYTE *v5; // rdx
  void (*v6)(); // rcx
  __int64 *v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  unsigned __int64 v10; // rax
  __int64 *v11; // rbx
  __int64 *v12; // rdi
  __int64 v13; // rax
  void (*v14)(); // rax
  __int64 v15; // [rsp+0h] [rbp-60h]
  _QWORD v16[2]; // [rsp+20h] [rbp-40h] BYREF
  __int16 v17; // [rsp+30h] [rbp-30h]

  if ( *(_BYTE *)(a2 + 416) )
  {
    v12 = *(__int64 **)(a2 + 256);
    v13 = *v12;
    LODWORD(v15) = *((_DWORD *)a1 + 2);
    v16[0] = "Size: ";
    v14 = *(void (**)())(v13 + 104);
    v16[1] = v15;
    v17 = 2307;
    if ( v14 != nullsub_580 )
      ((void (__fastcall *)(__int64 *, _QWORD *, __int64))v14)(v12, v16, 1);
  }
  switch ( a3 )
  {
    case 3:
      sub_396F320(a2, *((_DWORD *)a1 + 2));
      goto LABEL_4;
    case 4:
      sub_396F340(a2, *((_DWORD *)a1 + 2));
      if ( *(_BYTE *)(a2 + 416) )
        goto LABEL_11;
      goto LABEL_5;
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
    case 16:
    case 17:
    case 18:
    case 19:
    case 20:
    case 21:
    case 22:
    case 23:
    case 24:
      sub_397C0C0(a2, *((unsigned int *)a1 + 2), 0);
      goto LABEL_4;
    case 10:
      sub_396F300(a2, *((_DWORD *)a1 + 2));
LABEL_4:
      if ( *(_BYTE *)(a2 + 416) )
      {
LABEL_11:
        sub_3982BD0(a2, a1, v5, v6, v7, v8);
      }
      else
      {
LABEL_5:
        if ( *a1 )
        {
          v9 = **a1;
          do
          {
            v10 = v9 & 0xFFFFFFFFFFFFFFF8LL;
            v11 = (__int64 *)v10;
            if ( !v10 )
              break;
            sub_3982B10(v10 + 8, a2, (__int64)v5, (__int64)v6, (__int64)v7, v8);
            v9 = *v11;
          }
          while ( (*v11 & 4) == 0 );
        }
      }
      return;
  }
}
