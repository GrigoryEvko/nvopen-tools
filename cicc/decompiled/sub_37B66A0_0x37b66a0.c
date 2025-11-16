// Function: sub_37B66A0
// Address: 0x37b66a0
//
void __fastcall sub_37B66A0(unsigned int *a1, unsigned int *a2)
{
  unsigned int *v2; // r11
  unsigned int *v4; // r10
  unsigned int v5; // esi
  __int64 v6; // r10
  unsigned int v7; // ecx
  __int64 *v8; // rdi
  unsigned int *v9; // r12
  unsigned int v10; // r8d
  unsigned int v11; // r9d
  unsigned int v12; // r14d
  unsigned __int64 v13; // rax
  int v14; // edx

  if ( a1 != a2 )
  {
    v2 = a1;
    if ( a2 != a1 + 5 )
    {
      v4 = a1 + 10;
      while ( 1 )
      {
        v7 = *(v4 - 5);
        v8 = (__int64 *)(v4 - 5);
        v9 = v4;
        if ( v7 >= *v2 )
        {
          if ( v7 == *v2 )
          {
            v5 = *(v4 - 4);
            if ( v5 < v2[1] )
              goto LABEL_9;
          }
          sub_37B65B0(v8);
          v4 = (unsigned int *)(v6 + 20);
          if ( a2 == v9 )
            return;
        }
        else
        {
          v5 = *(v4 - 4);
LABEL_9:
          v10 = *(v4 - 3);
          v11 = *(v4 - 2);
          v12 = *(v4 - 1);
          v13 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v8 - (char *)v2) >> 2);
          if ( (char *)v8 - (char *)v2 > 0 )
          {
            do
            {
              v14 = *((_DWORD *)v8 - 5);
              v8 = (__int64 *)((char *)v8 - 20);
              *((_DWORD *)v8 + 5) = v14;
              *((_DWORD *)v8 + 6) = *((_DWORD *)v8 + 1);
              *((_DWORD *)v8 + 7) = *((_DWORD *)v8 + 2);
              *((_DWORD *)v8 + 8) = *((_DWORD *)v8 + 3);
              *((_DWORD *)v8 + 9) = *((_DWORD *)v8 + 4);
              --v13;
            }
            while ( v13 );
          }
          *v2 = v7;
          v4 += 5;
          v2[1] = v5;
          v2[2] = v10;
          v2[3] = v11;
          v2[4] = v12;
          if ( a2 == v9 )
            return;
        }
      }
    }
  }
}
