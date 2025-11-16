// Function: sub_2DF4310
// Address: 0x2df4310
//
__int64 __fastcall sub_2DF4310(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r14
  __int64 (*v3)(); // rdx
  __int64 v4; // rax
  __int64 (*v5)(); // rdx
  __int64 v6; // rax
  _QWORD *v7; // r13
  unsigned int v8; // r14d
  __int64 i; // rbx
  char v10; // al
  __int64 v11; // rax
  unsigned __int8 *v12; // rdx
  unsigned __int8 v13; // r10
  __int64 v14; // r14
  unsigned int v15; // r10d
  int v16; // eax
  unsigned __int8 v18; // [rsp+1Fh] [rbp-41h]
  __int64 v19[7]; // [rsp+28h] [rbp-38h] BYREF

  if ( !sub_BA91D0(*(_QWORD *)(*a2 + 40LL), "kcfi", 4u) )
    return 0;
  v2 = a2[2];
  v3 = *(__int64 (**)())(*(_QWORD *)v2 + 128LL);
  v4 = 0;
  if ( v3 != sub_2DAC790 )
    v4 = ((__int64 (__fastcall *)(_QWORD))v3)(a2[2]);
  *(_QWORD *)(a1 + 200) = v4;
  v5 = *(__int64 (**)())(*(_QWORD *)v2 + 144LL);
  v6 = 0;
  if ( v5 != sub_2C8F680 )
    v6 = ((__int64 (__fastcall *)(__int64))v5)(v2);
  *(_QWORD *)(a1 + 208) = v6;
  v7 = (_QWORD *)a2[41];
  if ( v7 == a2 + 40 )
  {
    return 0;
  }
  else
  {
    v8 = 0;
    do
    {
      for ( i = v7[7]; v7 + 6 != (_QWORD *)i; i = *(_QWORD *)(i + 8) )
      {
        v16 = *(_DWORD *)(i + 44);
        if ( (v16 & 4) != 0 || (v16 & 8) == 0 )
          v10 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(i + 16) + 24LL) >> 7;
        else
          v10 = sub_2E88A90(i, 128, 1);
        if ( v10 )
        {
          v11 = *(_QWORD *)(i + 48);
          v12 = (unsigned __int8 *)(v11 & 0xFFFFFFFFFFFFFFF8LL);
          if ( (v11 & 0xFFFFFFFFFFFFFFF8LL) != 0 && (v11 & 7) == 3 )
          {
            v13 = v12[8];
            if ( v13 )
            {
              if ( *(_DWORD *)&v12[8 * *(int *)v12 + 16 + 8 * v12[7] + 8 * v12[6] + 8 * (__int64)(v12[5] + v12[4])] )
              {
                v19[0] = i;
                if ( (*(_BYTE *)(i + 44) & 0xC) != 0 && *(_WORD *)((*(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL) + 68) != 21 )
                  sub_C64ED0("Cannot emit a KCFI check for a bundled call", 1u);
                v18 = v13;
                v14 = (*(__int64 (__fastcall **)(_QWORD, _QWORD *, __int64 *, _QWORD))(**(_QWORD **)(a1 + 208) + 1096LL))(
                        *(_QWORD *)(a1 + 208),
                        v7,
                        v19,
                        *(_QWORD *)(a1 + 200));
                sub_2E88490(v19[0], v7[4], 0);
                v15 = v18;
                if ( (*(_BYTE *)(v19[0] + 44) & 0xC) == 0 )
                {
                  sub_2E92D10(v7, v14, *(_QWORD *)(v19[0] + 8));
                  v15 = v18;
                }
                v8 = v15;
              }
            }
          }
        }
      }
      v7 = (_QWORD *)v7[1];
    }
    while ( a2 + 40 != v7 );
  }
  return v8;
}
