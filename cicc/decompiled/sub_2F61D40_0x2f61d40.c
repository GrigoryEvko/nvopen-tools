// Function: sub_2F61D40
// Address: 0x2f61d40
//
void __fastcall sub_2F61D40(char *a1, char *a2)
{
  char *i; // r11
  __int64 *v5; // rdi
  char *v6; // r12
  unsigned int v7; // edx
  unsigned int v8; // eax
  unsigned __int64 v9; // rcx
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdx

  if ( a1 != a2 && a2 != a1 + 16 )
  {
    for ( i = a1 + 32; ; i += 16 )
    {
      v5 = (__int64 *)(i - 16);
      v6 = i;
      v7 = *(_DWORD *)((*((_QWORD *)i - 2) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*((__int64 *)i - 2) >> 1) & 3;
      v8 = *(_DWORD *)((*(_QWORD *)a1 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*(__int64 *)a1 >> 1) & 3;
      if ( v7 < v8 )
        break;
      if ( v7 <= v8 )
      {
        v9 = *((_QWORD *)i - 1);
        if ( v9 < *((_QWORD *)a1 + 1) )
          goto LABEL_6;
      }
      sub_2F61B00(v5);
LABEL_9:
      if ( a2 == v6 )
        return;
    }
    v9 = *((_QWORD *)i - 1);
LABEL_6:
    v10 = *((_QWORD *)i - 2);
    v11 = ((char *)v5 - a1) >> 4;
    if ( (char *)v5 - a1 > 0 )
    {
      do
      {
        v12 = *(v5 - 2);
        v5 -= 2;
        v5[2] = v12;
        v5[3] = v5[1];
        --v11;
      }
      while ( v11 );
    }
    *(_QWORD *)a1 = v10;
    *((_QWORD *)a1 + 1) = v9;
    goto LABEL_9;
  }
}
