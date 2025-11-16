// Function: sub_246F3F0
// Address: 0x246f3f0
//
__int64 __fastcall sub_246F3F0(__int64 a1, __int64 a2)
{
  char v3; // al
  __int64 v4; // rsi
  _QWORD *v5; // rdi
  __int64 result; // rax
  __int64 v7; // rsi
  __int64 *v8; // rbx
  _QWORD *v9; // rdi
  __int64 v10; // r15
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rcx
  unsigned int v14; // r8d
  __int64 v15; // r11
  __int64 i; // r13
  __int64 v17; // rdx
  unsigned int v18; // eax
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rax
  unsigned __int16 v23; // ax
  unsigned int v24; // r11d
  unsigned __int16 v25; // dx
  __int64 v26; // rax
  _BYTE *v27; // r10
  __int64 v28; // rdx
  __int64 v29; // rax
  unsigned int v30; // eax
  __int64 v31; // rax
  __int64 v32; // rax
  int v33; // eax
  _QWORD *v34; // rax
  int v35; // ecx
  int v36; // edx
  __int64 v37; // rax
  __int64 v38; // rax
  unsigned __int64 v39; // rcx
  char *v40; // rax
  unsigned __int8 v41; // si
  unsigned int v42; // eax
  unsigned int v43; // edx
  unsigned __int64 v44; // rax
  unsigned __int16 v45; // cx
  unsigned int v46; // edx
  unsigned int v47; // [rsp+0h] [rbp-120h]
  __int64 v48; // [rsp+0h] [rbp-120h]
  __int64 v49; // [rsp+0h] [rbp-120h]
  unsigned int v50; // [rsp+8h] [rbp-118h]
  unsigned __int16 v51; // [rsp+8h] [rbp-118h]
  __int64 v52; // [rsp+8h] [rbp-118h]
  __int64 v53; // [rsp+8h] [rbp-118h]
  __int64 v54; // [rsp+10h] [rbp-110h]
  __int64 v55; // [rsp+10h] [rbp-110h]
  int v56; // [rsp+10h] [rbp-110h]
  unsigned int v57; // [rsp+20h] [rbp-100h]
  int v58; // [rsp+24h] [rbp-FCh]
  __int64 v59; // [rsp+28h] [rbp-F8h]
  __int64 v60; // [rsp+28h] [rbp-F8h]
  unsigned int v61; // [rsp+28h] [rbp-F8h]
  __int64 v62; // [rsp+28h] [rbp-F8h]
  unsigned __int8 v63; // [rsp+28h] [rbp-F8h]
  unsigned int v64; // [rsp+28h] [rbp-F8h]
  unsigned __int8 v65; // [rsp+28h] [rbp-F8h]
  _QWORD v66[4]; // [rsp+30h] [rbp-F0h] BYREF
  __int16 v67; // [rsp+50h] [rbp-D0h]
  unsigned int *v68[9]; // [rsp+60h] [rbp-C0h] BYREF
  _QWORD *v69; // [rsp+A8h] [rbp-78h]

  v3 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 > 0x1Cu )
  {
    if ( *(_BYTE *)(a1 + 633) && ((*(_BYTE *)(a2 + 7) & 0x20) == 0 || !sub_B91C10(a2, 31)) )
      return *sub_246EC10(a1 + 304, a2);
    v4 = *(_QWORD *)(a2 + 8);
    v5 = sub_2463540((__int64 *)a1, v4);
    if ( v5 )
      return sub_AD6530((__int64)v5, v4);
    return 0;
  }
  if ( (unsigned __int8)(v3 - 12) > 1u )
  {
    if ( v3 != 22 )
    {
      v7 = *(_QWORD *)(a2 + 8);
      return sub_24637B0((__int64 *)a1, v7);
    }
    v8 = sub_246EC10(a1 + 304, a2);
    result = *v8;
    if ( !*v8 )
    {
      v10 = *(_QWORD *)(a2 + 24);
      v11 = *(_QWORD *)(a1 + 480);
      sub_23D0AB0((__int64)v68, v11, 0, 0, 0);
      v54 = sub_B2BEC0(v10);
      if ( (*(_BYTE *)(v10 + 2) & 1) != 0 )
      {
        sub_B2C6D0(v10, v11, v12, v13);
        v15 = *(_QWORD *)(v10 + 96);
        v59 = v15 + 40LL * *(_QWORD *)(v10 + 104);
        if ( (*(_BYTE *)(v10 + 2) & 1) != 0 )
        {
          sub_B2C6D0(v10, v11, v12, v13);
          v15 = *(_QWORD *)(v10 + 96);
        }
      }
      else
      {
        v15 = *(_QWORD *)(v10 + 96);
        v59 = v15 + 40LL * *(_QWORD *)(v10 + 104);
      }
      v58 = 0;
      if ( v15 != v59 )
      {
        for ( i = v15; v59 != i; i += 40 )
        {
          v11 = 0;
          if ( !(unsigned __int8)sub_9C6430(*(_QWORD *)(i + 8), 0, v12, v13, v14) || sub_BCEA30(*(_QWORD *)(i + 8)) )
          {
            if ( i == a2 )
            {
              v19 = *(_QWORD *)(i + 8);
              *v8 = sub_24637B0((__int64 *)a1, v19);
              v20 = sub_AD6530(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 88LL), v19);
              v11 = i;
              sub_246F1C0(a1, i, v20);
              break;
            }
          }
          else
          {
            if ( (unsigned __int8)sub_B2D680(i) )
              v11 = sub_B2BD20(i);
            else
              v11 = *(_QWORD *)(i + 8);
            v66[0] = sub_BDB740(v54, v11);
            v66[1] = v17;
            v18 = sub_CA1930(v66);
            if ( i == a2 )
            {
              v61 = v18;
              v57 = v18 + v58;
              if ( (unsigned __int8)sub_B2D680(i) )
              {
                v50 = v61;
                v62 = sub_B2BD20(i);
                v23 = sub_B2BD00(i);
                v24 = v50;
                if ( !HIBYTE(v23) )
                {
                  LOBYTE(v23) = sub_AE5020(v54, v62);
                  v24 = v50;
                }
                v63 = v23;
                HIBYTE(v25) = 1;
                v47 = v24;
                LOBYTE(v25) = v23;
                LOBYTE(v66[0]) = v23;
                v51 = v25;
                v26 = sub_BCB2B0(v69);
                v27 = sub_2466120(a1, a2, v68, v26, v51, 1);
                v52 = v28;
                v55 = v47;
                if ( *(_BYTE *)(a1 + 633) == 1 && v57 <= 0x320 )
                {
                  v49 = (__int64)v27;
                  v39 = sub_24646E0(a1, (__int64)v68, v58);
                  v40 = &byte_4FE8EA8;
                  if ( v63 <= (unsigned __int8)byte_4FE8EA8 )
                    v40 = (char *)v66;
                  v41 = *v40;
                  v42 = 256;
                  LOBYTE(v42) = v41;
                  v43 = v41;
                  BYTE1(v43) = 1;
                  sub_2463B20((__int64)v68, v49, v43, v39, v42, v55, 0, 0, 0, 0, 0);
                  if ( *(_DWORD *)(*(_QWORD *)(a1 + 8) + 4LL) )
                  {
                    v44 = sub_24647C0(a1, (__int64)v68, v58);
                    LOBYTE(v45) = byte_4FE8EA9;
                    v46 = (unsigned __int8)byte_4FE8EA9;
                    BYTE1(v46) = 1;
                    HIBYTE(v45) = 1;
                    sub_2463B20(
                      (__int64)v68,
                      v52,
                      v46,
                      v44,
                      v45,
                      -(1 << byte_4FE8EA9) & ((unsigned int)(1LL << byte_4FE8EA9) + (unsigned int)v55 - 1),
                      0,
                      0,
                      0,
                      0,
                      0);
                  }
                }
                else
                {
                  v48 = (__int64)v27;
                  v29 = sub_BCB2B0(v69);
                  v53 = sub_AD6530(v29, a2);
                  v30 = v63;
                  BYTE1(v30) = 1;
                  v64 = v30;
                  v31 = sub_BCB2E0(v69);
                  v32 = sub_ACD640(v31, v55, 0);
                  sub_B34240((__int64)v68, v48, v53, v32, v64, 0, 0, 0, 0);
                }
              }
              if ( *(_BYTE *)(a1 + 633) != 1
                || v57 > 0x320
                || (unsigned __int8)sub_B2D680(i)
                || *(_BYTE *)(*(_QWORD *)(a1 + 8) + 9LL) && (unsigned __int8)sub_B2D670(i, 40) )
              {
                v21 = *(_QWORD *)(a2 + 8);
                *v8 = sub_24637B0((__int64 *)a1, v21);
                v22 = sub_AD6530(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 88LL), v21);
                v11 = a2;
                sub_246F1C0(a1, a2, v22);
              }
              else
              {
                v33 = sub_24646E0(a1, (__int64)v68, v58);
                v67 = 257;
                v56 = v33;
                v65 = byte_4FE8EA8;
                v34 = sub_2463540((__int64 *)a1, *(_QWORD *)(i + 8));
                v35 = v65;
                v11 = (__int64)v34;
                BYTE1(v35) = 1;
                *v8 = sub_A82CA0(v68, (__int64)v34, v56, v35, 0, (__int64)v66);
                if ( *(_DWORD *)(*(_QWORD *)(a1 + 8) + 4LL) )
                {
                  v36 = sub_24647C0(a1, (__int64)v68, v58);
                  v37 = *(_QWORD *)(a1 + 8);
                  v67 = 257;
                  v38 = sub_A82CA0(v68, *(_QWORD *)(v37 + 88), v36, 0, 0, (__int64)v66);
                  v11 = a2;
                  sub_246F1C0(a1, a2, v38);
                }
              }
              break;
            }
            v13 = (unsigned __int8)byte_4FE8EA8;
            v12 = (1LL << byte_4FE8EA8) + v18 - 1;
            v58 += v12 & -(1 << byte_4FE8EA8);
          }
        }
      }
      v60 = *v8;
      sub_F94A20(v68, v11);
      return v60;
    }
  }
  else
  {
    v7 = *(_QWORD *)(a2 + 8);
    if ( !*(_BYTE *)(a1 + 633) || !*(_BYTE *)(a1 + 635) )
      return sub_24637B0((__int64 *)a1, v7);
    v9 = sub_2463540((__int64 *)a1, v7);
    if ( !v9 )
      return 0;
    return sub_24623D0((__int64)v9);
  }
  return result;
}
