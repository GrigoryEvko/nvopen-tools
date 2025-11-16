// Function: sub_C92C20
// Address: 0xc92c20
//
__int64 __fastcall sub_C92C20(
        char *a1,
        unsigned __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        unsigned int a6)
{
  unsigned __int64 v9; // rax
  unsigned int *v10; // rax
  __int64 v11; // rsi
  _BYTE *v12; // rdi
  unsigned __int64 v13; // r13
  unsigned int *v14; // rdx
  __int64 v15; // rdx
  int v16; // eax
  unsigned __int64 v17; // r15
  unsigned int v18; // ecx
  unsigned __int64 v19; // rax
  unsigned int v20; // r15d
  unsigned int v21; // edi
  unsigned int v22; // r13d
  unsigned int v23; // edx
  __int64 v24; // rdx
  unsigned int v25; // r13d
  char v26; // r11
  char v27; // r10
  unsigned int *v28; // r9
  unsigned int v29; // r11d
  unsigned int v30; // r12d
  unsigned __int64 v34; // [rsp+18h] [rbp-148h]
  char v35; // [rsp+18h] [rbp-148h]
  unsigned int *v36; // [rsp+20h] [rbp-140h] BYREF
  __int64 v37; // [rsp+28h] [rbp-138h]
  _BYTE v38[304]; // [rsp+30h] [rbp-130h] BYREF

  if ( !a6 )
    goto LABEL_5;
  v9 = a4 - a2;
  if ( a2 > a4 )
    v9 = a2 - a4;
  if ( a6 < v9 )
  {
    return a6 + 1;
  }
  else
  {
LABEL_5:
    v10 = (unsigned int *)v38;
    v11 = 0x4000000000LL;
    v12 = v38;
    v36 = (unsigned int *)v38;
    v37 = 0x4000000000LL;
    v13 = a4 + 1;
    if ( a4 != -1 )
    {
      if ( v13 > 0x40 )
      {
        v11 = (__int64)v38;
        v35 = a5;
        sub_C8D5F0((__int64)&v36, v38, a4 + 1, 4u, a5, (__int64)&v36);
        v12 = v36;
        LOBYTE(a5) = v35;
        v10 = &v36[(unsigned int)v37];
      }
      v14 = (unsigned int *)&v12[4 * v13];
      if ( v14 != v10 )
      {
        do
        {
          if ( v10 )
            *v10 = 0;
          ++v10;
        }
        while ( v14 != v10 );
        v12 = v36;
      }
      LODWORD(v37) = v13;
      if ( (unsigned int)v13 > 1 )
      {
        v15 = 1;
        v16 = 1;
        do
        {
          *(_DWORD *)&v12[4 * v15] = v16;
          v15 = (unsigned int)(v16 + 1);
          v12 = v36;
          v16 = v15;
        }
        while ( (unsigned int)v37 > (unsigned int)v15 );
      }
    }
    v17 = 1;
    if ( a2 )
    {
      while ( 1 )
      {
        *(_DWORD *)v12 = v17;
        v12 = v36;
        v18 = v17 - 1;
        v19 = 1;
        v11 = *v36;
        if ( a4 )
        {
          v34 = v17;
          do
          {
            v24 = 4 * v19;
            v25 = v18;
            v26 = *a1;
            v27 = *(_BYTE *)(a3 + v19 - 1);
            v28 = (unsigned int *)&v12[4 * v19];
            v18 = *v28;
            if ( (_BYTE)a5 )
            {
              v20 = *v28;
              if ( *(_DWORD *)&v12[v24 - 4] <= v18 )
                v20 = *(_DWORD *)&v12[v24 - 4];
              v21 = v20 + 1;
              v22 = (v26 != v27) + v25;
              if ( v20 + 1 > v22 )
                v21 = v22;
              *v28 = v21;
            }
            else if ( v26 == v27 )
            {
              *v28 = v25;
            }
            else
            {
              v29 = *v28;
              if ( *(_DWORD *)&v12[v24 - 4] <= v18 )
                v29 = *(_DWORD *)&v12[v24 - 4];
              *v28 = v29 + 1;
            }
            v12 = v36;
            v23 = v36[v19];
            if ( (unsigned int)v11 > v23 )
              v11 = v23;
            ++v19;
          }
          while ( a4 >= v19 );
          v17 = v34;
        }
        if ( a6 && a6 < (unsigned int)v11 )
          break;
        ++v17;
        ++a1;
        if ( a2 < v17 )
          goto LABEL_37;
      }
      v30 = a6 + 1;
    }
    else
    {
LABEL_37:
      v30 = *(_DWORD *)&v12[4 * a4];
    }
    if ( v12 != v38 )
      _libc_free(v12, v11);
  }
  return v30;
}
