// Function: sub_7748D0
// Address: 0x7748d0
//
__int64 __fastcall sub_7748D0(__int64 a1, __int64 a2, __int64 a3, char **a4, __int64 a5)
{
  char *v6; // rdx
  char v7; // al
  __int64 v8; // rcx
  char v9; // al
  __int64 result; // rax
  __int64 v11; // rax
  __int64 v12; // rax

  v6 = *a4;
  v7 = **a4;
  if ( v7 != 48 )
  {
    switch ( v7 )
    {
      case 2:
        v12 = *((_QWORD *)v6 + 1);
        goto LABEL_19;
      case 6:
        v11 = *((_QWORD *)v6 + 1);
        goto LABEL_9;
      case 7:
      case 8:
        v11 = *(_QWORD *)(*((_QWORD *)v6 + 1) + 120LL);
        goto LABEL_9;
      case 11:
        v11 = *(_QWORD *)(*((_QWORD *)v6 + 1) + 152LL);
        goto LABEL_9;
      case 13:
        v11 = **((_QWORD **)v6 + 1);
        goto LABEL_9;
      case 37:
        v11 = *(_QWORD *)(*((_QWORD *)v6 + 1) + 40LL);
        goto LABEL_9;
      default:
        goto LABEL_6;
    }
  }
  v8 = *((_QWORD *)v6 + 1);
  v9 = *(_BYTE *)(v8 + 8);
  if ( v9 == 1 )
  {
    *v6 = 2;
    v12 = *(_QWORD *)(v8 + 32);
    *((_QWORD *)v6 + 1) = v12;
LABEL_19:
    v11 = *(_QWORD *)(v12 + 128);
  }
  else
  {
    if ( v9 == 2 )
    {
      *v6 = 59;
      *((_QWORD *)v6 + 1) = *(_QWORD *)(v8 + 32);
      goto LABEL_6;
    }
    if ( v9 )
      sub_721090();
    *v6 = 6;
    v11 = *(_QWORD *)(v8 + 32);
    *((_QWORD *)v6 + 1) = v11;
  }
LABEL_9:
  if ( v11 )
  {
    *(_QWORD *)(a5 + 8) = v11;
    *(_BYTE *)a5 = 6;
    *(_DWORD *)(a5 + 16) = 0;
    return 1;
  }
LABEL_6:
  result = 0;
  if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
  {
    sub_6855B0(0xD2Fu, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
    sub_770D30(a1);
    return 0;
  }
  return result;
}
