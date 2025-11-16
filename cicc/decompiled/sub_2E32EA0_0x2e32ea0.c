// Function: sub_2E32EA0
// Address: 0x2e32ea0
//
__int64 __fastcall sub_2E32EA0(__int64 a1, __int64 a2)
{
  unsigned int *v3; // rax
  unsigned int *v5; // r10
  unsigned int *v6; // rdi
  unsigned int *v7; // rax
  unsigned int v8; // edx
  int v9; // r8d
  unsigned int v10; // ecx
  __int64 v11; // rsi
  unsigned int v12; // eax
  unsigned int v13[3]; // [rsp+Ch] [rbp-14h] BYREF

  if ( *(_QWORD *)(a1 + 152) == *(_QWORD *)(a1 + 144) )
  {
    sub_F02DB0(v13, 1u, *(_DWORD *)(a1 + 120));
    return v13[0];
  }
  else
  {
    v3 = (unsigned int *)sub_2E32E80(a1, a2);
    if ( *v3 == -1 )
    {
      v5 = *(unsigned int **)(a1 + 144);
      v6 = *(unsigned int **)(a1 + 152);
      if ( v6 == v5 )
      {
        v12 = 0x80000000;
        v9 = 0;
      }
      else
      {
        v7 = *(unsigned int **)(a1 + 144);
        v8 = 0;
        v9 = 0;
        do
        {
          v10 = *v7;
          if ( *v7 != -1 )
          {
            v11 = v8;
            v8 += v10;
            if ( (unsigned __int64)v10 + v11 > 0x80000000 )
              v8 = 0x80000000;
            ++v9;
          }
          ++v7;
        }
        while ( v6 != v7 );
        v12 = 0x80000000 - v8;
      }
      return v12 / ((unsigned int)(v6 - v5) - v9);
    }
    else
    {
      return *v3;
    }
  }
}
