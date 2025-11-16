// Function: sub_2A4C510
// Address: 0x2a4c510
//
void __fastcall sub_2A4C510(__int64 a1, unsigned __int8 *a2, unsigned __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v9; // r12
  __int64 v10; // r13
  __int64 **v11; // rax
  __int64 v12; // r12
  _QWORD *v13; // rdi
  __int64 v14; // r9
  __int64 *v15; // rax
  __int64 v16; // r14
  __int64 v17; // rdx
  __int64 v18; // r15
  int v19; // ecx
  int v20; // eax
  _QWORD *v21; // rdi
  __int64 *v22; // rax
  __int64 v23; // rsi
  __int64 v24; // r10
  _QWORD *v25; // r15
  __int64 v26; // r11
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 *v30; // r9
  __int64 v31; // [rsp+10h] [rbp-A0h]
  __int64 v32; // [rsp+10h] [rbp-A0h]
  _QWORD *v33; // [rsp+18h] [rbp-98h]
  _QWORD *v34; // [rsp+20h] [rbp-90h] BYREF
  __int64 v35; // [rsp+28h] [rbp-88h]
  __m128i v36; // [rsp+30h] [rbp-80h] BYREF
  __int64 v37; // [rsp+40h] [rbp-70h]
  __int64 v38; // [rsp+48h] [rbp-68h]
  __int64 v39; // [rsp+50h] [rbp-60h]
  __int64 v40; // [rsp+58h] [rbp-58h]
  __int64 v41; // [rsp+60h] [rbp-50h]
  __int64 v42; // [rsp+68h] [rbp-48h]
  __int16 v43; // [rsp+70h] [rbp-40h]

  if ( (unsigned int)*a2 - 12 > 1 )
    goto LABEL_27;
  if ( (*(_BYTE *)(a1 + 7) & 0x20) == 0 )
    return;
  if ( sub_B91C10(a1, 29) )
  {
    v9 = (__int64 *)sub_BD5C60(a1);
    v10 = sub_ACD6D0(v9);
    v11 = (__int64 **)sub_BCE3C0(v9, 0);
    v12 = sub_ACADE0(v11);
    v13 = sub_BD2C40(80, unk_3F10A10);
    if ( v13 )
      sub_B4D3C0((__int64)v13, v10, v12, 0, 0, v14, a1 + 24, 0);
  }
  else
  {
LABEL_27:
    if ( a4 )
    {
      if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
      {
        if ( sub_B91C10(a1, 11) )
        {
          if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
          {
            if ( sub_B91C10(a1, 29) )
            {
              v36 = (__m128i)a3;
              v43 = 257;
              v37 = 0;
              v38 = a5;
              v39 = a4;
              v40 = a1;
              v41 = 0;
              v42 = 0;
              if ( !(unsigned __int8)sub_9B6260((__int64)a2, &v36, 0) )
              {
                v15 = (__int64 *)sub_B43CA0(a1);
                v16 = sub_B6E160(v15, 0xBu, 0, 0);
                v31 = sub_AD6530(*(_QWORD *)(a1 + 8), 11);
                LOWORD(v39) = 257;
                v33 = sub_BD2C40(72, unk_3F10FD0);
                if ( v33 )
                {
                  v17 = *(_QWORD *)(a1 + 8);
                  v18 = (__int64)v33;
                  v19 = *(unsigned __int8 *)(v17 + 8);
                  if ( (unsigned int)(v19 - 17) > 1 )
                  {
                    v23 = sub_BCB2A0(*(_QWORD **)v17);
                  }
                  else
                  {
                    v20 = *(_DWORD *)(v17 + 32);
                    v21 = *(_QWORD **)v17;
                    BYTE4(v35) = (_BYTE)v19 == 18;
                    LODWORD(v35) = v20;
                    v22 = (__int64 *)sub_BCB2A0(v21);
                    v23 = sub_BCE1B0(v22, v35);
                  }
                  sub_B523C0((__int64)v33, v23, 53, 33, a1, v31, (__int64)&v36, 0, 0, 0);
                }
                else
                {
                  v18 = 0;
                }
                sub_B43E90(v18, a1 + 24);
                v24 = 0;
                LOWORD(v39) = 257;
                v34 = v33;
                if ( v16 )
                  v24 = *(_QWORD *)(v16 + 24);
                v32 = v24;
                v25 = sub_BD2CC0(88, 2u);
                if ( v25 )
                {
                  sub_B44260((__int64)v25, **(_QWORD **)(v32 + 16), 56, 2u, 0, 0);
                  v25[9] = 0;
                  sub_B4A290((__int64)v25, v32, v16, (__int64 *)&v34, 1, (__int64)&v36, 0, 0);
                  v26 = (__int64)v25;
                }
                else
                {
                  v26 = 0;
                }
                sub_B43E90(v26, (__int64)(v33 + 3));
                sub_CFEAE0(a4, (__int64)v25, v27, v28, v29, v30);
              }
            }
          }
        }
      }
    }
  }
}
