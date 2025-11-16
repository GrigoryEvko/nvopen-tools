// Function: sub_689CB0
// Address: 0x689cb0
//
__int64 __fastcall sub_689CB0(__int64 *a1, __int64 *a2)
{
  unsigned int v2; // r15d
  __int64 v3; // r14
  __int64 v5; // rbx
  char v6; // r12
  char *v7; // r8
  char v8; // al
  __int64 v9; // rdx
  __int64 v11; // rdx
  __int64 v12; // rdi
  int v13; // eax
  __int64 v14; // rcx
  __int64 v15; // r9
  __int64 v16; // rdx
  int v17; // eax
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // r15
  _QWORD *v25; // rax
  __int64 v26; // rdx
  _QWORD *v27; // r12
  char *v28; // [rsp+0h] [rbp-1C0h]
  __int64 v29; // [rsp+0h] [rbp-1C0h]
  __int64 v30; // [rsp+8h] [rbp-1B8h]
  char *v31; // [rsp+8h] [rbp-1B8h]
  unsigned int v32; // [rsp+1Ch] [rbp-1A4h] BYREF
  __int64 v33; // [rsp+20h] [rbp-1A0h] BYREF
  __int64 v34; // [rsp+28h] [rbp-198h] BYREF
  char v35; // [rsp+30h] [rbp-190h] BYREF

  v2 = 0;
  v3 = 0;
  v5 = unk_4F06218;
  v6 = *(_BYTE *)(unk_4F06218 + 80LL);
  sub_7296C0(&v32);
  if ( v6 != 17 || (v5 = *(_QWORD *)(v5 + 88)) != 0 )
  {
    v7 = &v35;
    while ( 1 )
    {
      v8 = *(_BYTE *)(v5 + 80);
      v9 = v5;
      if ( v8 == 16 )
      {
        v9 = **(_QWORD **)(v5 + 88);
        v8 = *(_BYTE *)(v9 + 80);
      }
      if ( v8 == 24 )
      {
        v9 = *(_QWORD *)(v9 + 88);
        v8 = *(_BYTE *)(v9 + 80);
      }
      if ( v8 == 20 && dword_4F077C4 == 2 )
      {
        v11 = **(_QWORD **)(*(_QWORD *)(v9 + 88) + 328LL);
        if ( unk_4F07778 > 202001 )
        {
          if ( v11 )
          {
            if ( !*(_QWORD *)v11 && (*(_BYTE *)(v11 + 56) & 0x10) == 0 && *(_BYTE *)(*(_QWORD *)(v11 + 8) + 80LL) == 2 )
            {
              v28 = v7;
              v30 = v11;
              v12 = *(_QWORD *)(*(_QWORD *)(v11 + 64) + 128LL);
              v13 = sub_8D3A70(v12);
              v16 = v30;
              v7 = v28;
              if ( v13
                || (v12 = *(_QWORD *)(*(_QWORD *)(v30 + 64) + 128LL), v17 = sub_8D3F60(v12), v16 = v30, v7 = v28, v17) )
              {
                v29 = v16;
                v31 = v7;
                v33 = sub_724DC0(v12, a2, v16, v14, v7, v15);
                sub_6E6A50(&unk_4F06220, v31);
                a2 = *(__int64 **)(*(_QWORD *)(v29 + 64) + 128LL);
                if ( (unsigned int)sub_84D340(v31, a2, v33) )
                {
                  v18 = sub_725090(1);
                  *(_BYTE *)(v18 + 24) &= ~8u;
                  v34 = v18;
                  *(_QWORD *)(v18 + 32) = sub_724E50(&v33, a2, v19, v20, v21);
                  sub_6E6A50(&unk_4F06220, v31);
                  v22 = *(_QWORD *)(v34 + 32);
                  *(_QWORD *)(v22 + 144) = sub_6F6F40(v31, 0);
                  a2 = &v34;
                  v23 = sub_8B74F0(v5, &v34, 1, &dword_4F063F8);
                  v24 = v23;
                  if ( v3 )
                  {
                    v25 = sub_67D9D0(0x9B7u, &dword_4F063F8);
                    v26 = v3;
                    v3 = 0;
                    v27 = v25;
                    sub_67E1D0(v25, 421, v26);
                    sub_67E1D0(v27, 421, v24);
                    v2 = 1;
                    sub_685910((__int64)v27, (FILE *)0x1A5);
                    break;
                  }
                  v7 = v31;
                  v3 = v23;
                }
                else
                {
                  sub_724E30(&v33);
                  v7 = v31;
                }
                v2 = 1;
              }
            }
          }
        }
      }
      if ( v6 == 17 )
      {
        v5 = *(_QWORD *)(v5 + 8);
        if ( v5 )
          continue;
      }
      break;
    }
  }
  *a1 = v3;
  sub_729730(v32);
  return v2;
}
