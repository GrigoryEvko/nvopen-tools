// Function: sub_9B3D40
// Address: 0x9b3d40
//
void __fastcall sub_9B3D40(__int64 a1, __int64 *a2, __int16 a3, int *a4, int a5, __m128i *a6)
{
  __int64 v6; // rbp
  unsigned int v7; // r8d
  __int64 **v9; // rdi
  char v10; // al
  int v11; // edx
  __int16 v12; // ax
  int v13; // eax
  _QWORD v14[4]; // [rsp-20h] [rbp-20h] BYREF

  if ( (a3 & 0x1F) != 0 )
  {
    v14[3] = v6;
    v7 = a5 + 1;
    v14[0] = 1023;
    if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
      v9 = *(__int64 ***)(a1 - 8);
    else
      v9 = (__int64 **)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
    sub_9B0780(*v9, a2, a3, (unsigned int *)v14, v7, a6);
    v10 = v14[0];
    if ( (v14[0] & 0x1C) == 0 )
    {
      v11 = *a4;
      *a4 &= 0x3E3u;
      if ( (v11 & 3) == 0 && !*((_BYTE *)a4 + 5) )
      {
        if ( (v11 & 0x20) != 0 )
        {
          if ( (v11 & 0x3C0) == 0 )
            *((_WORD *)a4 + 2) = 257;
        }
        else
        {
          *((_WORD *)a4 + 2) = 256;
        }
      }
    }
    if ( (v10 & 3) != 0 )
    {
      if ( (v10 & 1) == 0 )
      {
        v13 = *a4;
        *a4 &= 0x3FEu;
        if ( (v13 & 2) == 0 && !*((_BYTE *)a4 + 5) )
        {
          if ( (v13 & 0x3C) != 0 )
          {
            if ( (v13 & 0x3C0) == 0 )
              *((_WORD *)a4 + 2) = 257;
          }
          else
          {
            *((_WORD *)a4 + 2) = 256;
          }
        }
      }
    }
    else
    {
      v12 = WORD2(v14[0]);
      *a4 &= 0x3FCu;
      *((_WORD *)a4 + 2) = v12;
    }
  }
}
