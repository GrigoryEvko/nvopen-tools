// Function: sub_2522A30
// Address: 0x2522a30
//
__int64 __fastcall sub_2522A30(
        __int64 a1,
        __int64 a2,
        __int64 (__fastcall *a3)(__int64, unsigned __int64, __int64),
        __int64 a4,
        __int64 a5,
        _QWORD *a6,
        int *a7,
        __int64 a8,
        _BYTE *a9,
        char a10,
        unsigned __int8 a11)
{
  int v13; // ecx
  __int64 v14; // rsi
  int v15; // eax
  int v16; // edi
  unsigned int v17; // edx
  int *v18; // rax
  int v19; // r8d
  __int64 v20; // rax
  __int64 v21; // r12
  unsigned __int64 *i; // r15
  __int64 result; // rax
  unsigned __int64 v24; // r11
  int v25; // eax
  unsigned __int64 v26; // rax
  __int64 v27; // rcx
  unsigned __int64 v28; // rax
  int v29; // eax
  int v30; // r9d
  __int64 v31; // [rsp-10h] [rbp-A0h]
  int *v32; // [rsp+8h] [rbp-88h]
  unsigned __int64 v37; // [rsp+30h] [rbp-60h]
  int *v38; // [rsp+48h] [rbp-48h]
  __int64 v39[8]; // [rsp+50h] [rbp-40h] BYREF

  v38 = a7;
  v32 = &a7[a8];
  if ( a7 != v32 )
  {
    do
    {
      v13 = *v38;
      v14 = *(_QWORD *)(a2 + 8);
      v15 = *(_DWORD *)(a2 + 24);
      if ( v15 )
      {
        v16 = v15 - 1;
        v17 = (v15 - 1) & (37 * v13);
        v18 = (int *)(v14 + 16LL * v17);
        v19 = *v18;
        if ( v13 == *v18 )
        {
LABEL_4:
          v20 = *((_QWORD *)v18 + 1);
          if ( v20 )
          {
            v21 = *(_QWORD *)v20 + 8LL * *(unsigned int *)(v20 + 8);
            if ( v21 != *(_QWORD *)v20 )
            {
              for ( i = *(unsigned __int64 **)v20; (unsigned __int64 *)v21 != i; ++i )
              {
                v24 = *i;
                if ( ((a11 ^ 1) & (a1 != 0)) != 0 )
                {
                  v39[0] = 0;
                  v39[1] = 0;
                  v25 = *(unsigned __int8 *)v24;
                  if ( (_BYTE)v25
                    && ((unsigned __int8)v25 <= 0x1Cu
                     || (v26 = (unsigned int)(v25 - 34), (unsigned __int8)v26 > 0x33u)
                     || (v27 = 0x8000000000041LL, !_bittest64(&v27, v26))) )
                  {
                    v28 = v24 & 0xFFFFFFFFFFFFFFFCLL;
                  }
                  else
                  {
                    v28 = v24 & 0xFFFFFFFFFFFFFFFCLL | 2;
                  }
                  v37 = v24;
                  v39[0] = v28;
                  nullsub_1518();
                  if ( !(unsigned __int8)sub_251C230(a1, v39, a5, a6, a9, a10, 1) )
                  {
                    result = a3(a4, v37, v31);
                    if ( !(_BYTE)result )
                      return result;
                  }
                }
                else
                {
                  result = ((__int64 (__fastcall *)(__int64, unsigned __int64))a3)(a4, *i);
                  if ( !(_BYTE)result )
                    return result;
                }
              }
            }
          }
        }
        else
        {
          v29 = 1;
          while ( v19 != -1 )
          {
            v30 = v29 + 1;
            v17 = v16 & (v29 + v17);
            v18 = (int *)(v14 + 16LL * v17);
            v19 = *v18;
            if ( v13 == *v18 )
              goto LABEL_4;
            v29 = v30;
          }
        }
      }
      ++v38;
    }
    while ( v32 != v38 );
  }
  return 1;
}
