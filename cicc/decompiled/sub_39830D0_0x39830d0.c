// Function: sub_39830D0
// Address: 0x39830d0
//
void __fastcall sub_39830D0(__int64 **a1, __int64 a2, __int64 a3, void (*a4)(), __int64 *a5, __int64 a6)
{
  __int64 v6; // r15
  __int16 v8; // bx
  __int64 v9; // rax
  unsigned __int64 v10; // rax
  __int64 *v11; // rbx
  __int64 *v12; // rdi
  __int64 v13; // rax
  void (*v14)(); // rax
  __int64 v15; // [rsp+0h] [rbp-60h]
  _QWORD v16[2]; // [rsp+20h] [rbp-40h] BYREF
  __int16 v17; // [rsp+30h] [rbp-30h]

  v8 = a3;
  if ( !*(_BYTE *)(a2 + 416) )
  {
    switch ( (__int16)a3 )
    {
      case 3:
        goto LABEL_11;
      case 4:
        goto LABEL_8;
      case 8:
      case 30:
        goto LABEL_3;
      case 9:
        goto LABEL_13;
      case 10:
        goto LABEL_12;
      default:
        ++*(_DWORD *)(v6 + 6568);
        BUG();
    }
  }
  v12 = *(__int64 **)(a2 + 256);
  a4 = (void (*)())"Size: ";
  v13 = *v12;
  LODWORD(v15) = *((_DWORD *)a1 + 2);
  v16[0] = "Size: ";
  v14 = *(void (**)())(v13 + 104);
  v16[1] = v15;
  a3 = 2307;
  v17 = 2307;
  if ( v14 != nullsub_580 )
    ((void (__fastcall *)(__int64 *, _QWORD *, __int64))v14)(v12, v16, 1);
  switch ( v8 )
  {
    case 3:
LABEL_11:
      sub_396F320(a2, *((_DWORD *)a1 + 2));
      break;
    case 4:
LABEL_8:
      sub_396F340(a2, *((_DWORD *)a1 + 2));
      break;
    case 5:
    case 6:
    case 7:
    case 8:
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
    case 25:
    case 26:
    case 27:
    case 28:
    case 29:
    case 30:
      break;
    case 9:
LABEL_13:
      sub_397C0C0(a2, *((unsigned int *)a1 + 2), 0);
      break;
    case 10:
LABEL_12:
      sub_396F300(a2, *((_DWORD *)a1 + 2));
      break;
  }
  if ( *(_BYTE *)(a2 + 416) )
  {
    sub_3982BD0(a2, a1, (_BYTE *)a3, a4, a5, a6);
  }
  else
  {
LABEL_3:
    if ( *a1 )
    {
      v9 = **a1;
      do
      {
        v10 = v9 & 0xFFFFFFFFFFFFFFF8LL;
        v11 = (__int64 *)v10;
        if ( !v10 )
          break;
        sub_3982B10(v10 + 8, a2, a3, (__int64)a4, (__int64)a5, a6);
        v9 = *v11;
      }
      while ( (*v11 & 4) == 0 );
    }
  }
}
