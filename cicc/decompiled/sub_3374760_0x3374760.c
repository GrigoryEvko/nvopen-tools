// Function: sub_3374760
// Address: 0x3374760
//
__int64 __fastcall sub_3374760(__int64 a1, __int64 *a2, __int64 a3, int a4, __int64 a5, int a6)
{
  int v6; // r15d
  unsigned int v10; // eax
  unsigned int v11; // r9d
  __int64 *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // r8
  unsigned int v18; // ecx
  __int64 *v19; // rdx
  __int64 v20; // r11
  int v21; // esi
  __int64 v22; // rdx
  _DWORD *v23; // rax
  _DWORD *v24; // rdi
  unsigned int v25; // r14d
  __int64 v26; // rax
  int v27; // edx
  int v28; // r14d

  v6 = a5;
  LOBYTE(v10) = sub_AF46F0(a5);
  v11 = v10;
  if ( (_BYTE)v10 )
  {
    v13 = &a2[a3];
    if ( a2 != v13 && v13 == a2 + 1 )
    {
      v14 = *(_QWORD *)(a1 + 960);
      v15 = *a2;
      v16 = *(unsigned int *)(v14 + 144);
      v17 = *(_QWORD *)(v14 + 128);
      if ( (_DWORD)v16 )
      {
        v18 = (v16 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
        v19 = (__int64 *)(v17 + 16LL * v18);
        v20 = *v19;
        if ( v15 == *v19 )
        {
LABEL_8:
          if ( v19 != (__int64 *)(v17 + 16 * v16) )
          {
            v21 = *((_DWORD *)v19 + 2);
            v22 = *(_QWORD *)(v14 + 24);
            v23 = *(_DWORD **)(v22 + 488);
            v24 = *(_DWORD **)(v22 + 496);
            if ( v23 != v24 )
            {
              while ( 1 )
              {
                LOBYTE(v18) = *v23 == v21 || v23[1] == v21;
                v25 = v18;
                if ( (_BYTE)v18 )
                  break;
                v23 += 2;
                if ( v24 == v23 )
                  return v11;
              }
              v26 = sub_33E5F30(*(_QWORD *)(a1 + 864), a4, v6, *v23, 0, a6, *(_DWORD *)(a1 + 848));
              sub_33F99B0(*(_QWORD *)(a1 + 864), v26, 0);
              return v25;
            }
          }
        }
        else
        {
          v27 = 1;
          while ( v20 != -4096 )
          {
            v28 = v27 + 1;
            v18 = (v16 - 1) & (v27 + v18);
            v19 = (__int64 *)(v17 + 16LL * v18);
            v20 = *v19;
            if ( v15 == *v19 )
              goto LABEL_8;
            v27 = v28;
          }
        }
      }
    }
    else
    {
      return 0;
    }
  }
  return v11;
}
