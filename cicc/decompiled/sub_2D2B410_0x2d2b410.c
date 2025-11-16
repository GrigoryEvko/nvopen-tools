// Function: sub_2D2B410
// Address: 0x2d2b410
//
char __fastcall sub_2D2B410(unsigned int *a1, unsigned int a2)
{
  unsigned int *v2; // rax
  unsigned int *v3; // rbx
  int *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rsi
  int v8; // eax
  unsigned int v9; // r8d
  unsigned int v10; // ebx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9

  v2 = (unsigned int *)sub_2D289F0((__int64)a1);
  v3 = v2;
  if ( *v2 > a2 )
  {
    v4 = (int *)sub_2D28A30((__int64)a1);
    LOBYTE(v2) = sub_2D28DB0((__int64)a1, a2, *v4);
    if ( (_BYTE)v2 )
    {
      v5 = *((_QWORD *)a1 + 1);
      v6 = a1[4];
      v7 = v5 + 16 * v6 - 16;
      v8 = *(_DWORD *)(v7 + 12);
      if ( v8 )
      {
        if ( (_DWORD)v6 && *(_DWORD *)(v5 + 12) < *(_DWORD *)(v5 + 8) || (v9 = *(_DWORD *)(*(_QWORD *)a1 + 192LL)) == 0 )
        {
          *(_DWORD *)(v7 + 12) = v8 - 1;
LABEL_11:
          v10 = *(_DWORD *)sub_2D289F0((__int64)a1);
          sub_2D2B2D0((__int64)a1, v7, v11, v12, v13, v14);
          v2 = (unsigned int *)sub_2D289F0((__int64)a1);
          *v2 = v10;
          return (char)v2;
        }
      }
      else
      {
        v9 = *(_DWORD *)(*(_QWORD *)a1 + 192LL);
      }
      v7 = v9;
      sub_F03AD0(a1 + 2, v9);
      goto LABEL_11;
    }
  }
  *v3 = a2;
  return (char)v2;
}
