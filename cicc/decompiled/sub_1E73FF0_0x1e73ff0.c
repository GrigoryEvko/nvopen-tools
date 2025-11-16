// Function: sub_1E73FF0
// Address: 0x1e73ff0
//
void __fastcall sub_1E73FF0(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  __int64 v6; // rdi
  int v9; // r15d
  int v10; // eax
  __int64 v11; // rax
  int v12; // r15d
  int v13; // eax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rcx
  int v18; // r15d
  int v19; // eax
  __int64 v20; // rsi
  unsigned int v21; // edx
  unsigned int v22; // eax
  __int64 v23; // rax
  __int64 v24; // rax

  v6 = *(_QWORD *)(a2 + 16);
  if ( !v6 )
    goto LABEL_25;
  v9 = sub_1E73F70(v6, *(_BYTE *)(a2 + 25));
  v10 = sub_1E73F70(*(_QWORD *)(a3 + 16), *(_BYTE *)(a3 + 25));
  if ( !(unsigned __int8)sub_1E738F0(v10, v9, a3, a2, 2u) )
  {
    v11 = *(_QWORD *)(a1 + 128);
    if ( !*(_BYTE *)(v11 + 2568)
      || !(unsigned __int8)sub_1E73DD0(
                             (_WORD *)(a3 + 26),
                             (__int16 *)(a2 + 26),
                             a3,
                             a2,
                             3u,
                             *(_QWORD *)(a1 + 24),
                             *(_QWORD *)(v11 + 32))
      && ((v23 = *(_QWORD *)(a1 + 128), !*(_BYTE *)(v23 + 2568))
       || !(unsigned __int8)sub_1E73DD0(
                              (_WORD *)(a3 + 30),
                              (__int16 *)(a2 + 30),
                              a3,
                              a2,
                              4u,
                              *(_QWORD *)(a1 + 24),
                              *(_QWORD *)(v23 + 32))) )
    {
      if ( !a4
        || (!*(_BYTE *)(a1 + 44) || a4[42] || !(unsigned __int8)sub_1E73920(a3, a2, a4))
        && (v12 = sub_1E72BB0((__int64)a4, *(_QWORD *)(a2 + 16)),
            v13 = sub_1E72BB0((__int64)a4, *(_QWORD *)(a3 + 16)),
            !(unsigned __int8)sub_1E738C0(v13, v12, a3, a2, 5u)) )
      {
        v14 = *(_QWORD *)(a1 + 128);
        v15 = *(_QWORD *)(v14 + 2256);
        v16 = *(_QWORD *)(v14 + 2264);
        v17 = v15;
        if ( *(_BYTE *)(a2 + 25) )
          v17 = v16;
        if ( !*(_BYTE *)(a3 + 25) )
          v16 = v15;
        if ( !(unsigned __int8)sub_1E738F0(*(_QWORD *)(a3 + 16) == v16, *(_QWORD *)(a2 + 16) == v17, a3, a2, 6u) )
        {
          if ( !a4 )
          {
            v24 = *(_QWORD *)(a1 + 128);
            if ( *(_BYTE *)(v24 + 2568) )
              sub_1E73DD0(
                (_WORD *)(a3 + 34),
                (__int16 *)(a2 + 34),
                a3,
                a2,
                8u,
                *(_QWORD *)(a1 + 24),
                *(_QWORD *)(v24 + 32));
            return;
          }
          v18 = sub_1E73F50(*(_QWORD *)(a2 + 16), *(_BYTE *)(a2 + 25));
          v19 = sub_1E73F50(*(_QWORD *)(a3 + 16), *(_BYTE *)(a3 + 25));
          if ( (unsigned __int8)sub_1E738C0(v19, v18, a3, a2, 7u) )
            return;
          v20 = *(_QWORD *)(a1 + 128);
          if ( *(_BYTE *)(v20 + 2568) )
          {
            if ( (unsigned __int8)sub_1E73DD0(
                                    (_WORD *)(a3 + 34),
                                    (__int16 *)(a2 + 34),
                                    a3,
                                    a2,
                                    8u,
                                    *(_QWORD *)(a1 + 24),
                                    *(_QWORD *)(v20 + 32)) )
              return;
            v20 = *(_QWORD *)(a1 + 128);
          }
          sub_1E736C0(a3, v20, *(_QWORD *)(a1 + 16));
          if ( !(unsigned __int8)sub_1E738C0(*(_DWORD *)(a3 + 40), *(_DWORD *)(a2 + 40), a3, a2, 9u)
            && !(unsigned __int8)sub_1E738F0(*(_DWORD *)(a3 + 44), *(_DWORD *)(a2 + 44), a3, a2, 0xAu)
            && (*(_BYTE *)(a1 + 140) || !*(_BYTE *)a3
                                     || *(_BYTE *)(a1 + 44)
                                     || !(unsigned __int8)sub_1E73920(a3, a2, a4)) )
          {
            v21 = *(_DWORD *)(*(_QWORD *)(a3 + 16) + 192LL);
            v22 = *(_DWORD *)(*(_QWORD *)(a2 + 16) + 192LL);
            if ( a4[6] == 1 )
            {
              if ( v22 <= v21 )
                return;
            }
            else if ( v22 >= v21 )
            {
              return;
            }
LABEL_25:
            *(_BYTE *)(a3 + 24) = 16;
          }
        }
      }
    }
  }
}
