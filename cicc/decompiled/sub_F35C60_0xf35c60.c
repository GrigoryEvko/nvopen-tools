// Function: sub_F35C60
// Address: 0xf35c60
//
__int64 __fastcall sub_F35C60(__int64 a1, __int64 *a2)
{
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdi
  unsigned __int8 *v10; // rbx
  __int64 (__fastcall *v11)(__int64, __int64, __int64, unsigned __int8 *, __int64, __int64, const char *, __int64, const char *); // rax
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v16; // rbx
  __int64 v17; // r14
  __int64 v18; // rdx
  unsigned int v19; // esi
  const char *v20; // [rsp+0h] [rbp-90h] BYREF
  __int64 v21; // [rsp+8h] [rbp-88h]
  const char *v22; // [rsp+10h] [rbp-80h]
  __int16 v23; // [rsp+20h] [rbp-70h]
  _BYTE v24[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v25; // [rsp+50h] [rbp-40h]

  v3 = *(_QWORD *)(a1 - 96);
  v4 = *(_QWORD *)(v3 + 16);
  if ( !v4 || *(_QWORD *)(v4 + 8) || (unsigned __int8)(*(_BYTE *)v3 - 82) > 1u )
  {
    v20 = sub_BD5D20(*(_QWORD *)(a1 - 96));
    v23 = 773;
    v21 = v5;
    v22 = ".not";
    v6 = sub_AD62B0(*(_QWORD *)(v3 + 8));
    v9 = a2[10];
    v10 = (unsigned __int8 *)v6;
    v11 = *(__int64 (__fastcall **)(__int64, __int64, __int64, unsigned __int8 *, __int64, __int64, const char *, __int64, const char *))(*(_QWORD *)v9 + 16LL);
    if ( (char *)v11 == (char *)sub_9202E0 )
    {
      if ( *(_BYTE *)v3 > 0x15u || *v10 > 0x15u )
      {
LABEL_16:
        v25 = 257;
        v12 = sub_B504D0(30, v3, (__int64)v10, (__int64)v24, 0, 0);
        (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64))(*(_QWORD *)a2[11] + 16LL))(
          a2[11],
          v12,
          &v20,
          a2[7],
          a2[8]);
        v16 = *a2;
        v17 = *a2 + 16LL * *((unsigned int *)a2 + 2);
        if ( *a2 != v17 )
        {
          do
          {
            v18 = *(_QWORD *)(v16 + 8);
            v19 = *(_DWORD *)v16;
            v16 += 16;
            sub_B99FD0(v12, v19, v18);
          }
          while ( v17 != v16 );
        }
        if ( !*(_QWORD *)(a1 - 96) || (v13 = *(_QWORD *)(a1 - 88), (**(_QWORD **)(a1 - 80) = v13) == 0) )
        {
LABEL_11:
          *(_QWORD *)(a1 - 96) = v12;
          if ( !v12 )
            return sub_B4CC70(a1);
          goto LABEL_12;
        }
LABEL_10:
        *(_QWORD *)(v13 + 16) = *(_QWORD *)(a1 - 80);
        goto LABEL_11;
      }
      if ( (unsigned __int8)sub_AC47B0(30) )
        v12 = sub_AD5570(30, v3, v10, 0, 0);
      else
        v12 = sub_AABE40(0x1Eu, (unsigned __int8 *)v3, v10);
    }
    else
    {
      v12 = v11(v9, 30, v3, v10, v7, v8, v20, v21, v22);
    }
    if ( v12 )
    {
      if ( !*(_QWORD *)(a1 - 96) )
        goto LABEL_25;
      v13 = *(_QWORD *)(a1 - 88);
      **(_QWORD **)(a1 - 80) = v13;
      if ( !v13 )
        goto LABEL_25;
      goto LABEL_10;
    }
    goto LABEL_16;
  }
  *(_WORD *)(v3 + 2) = sub_B52870(*(_WORD *)(v3 + 2) & 0x3F) | *(_WORD *)(v3 + 2) & 0xFFC0;
  if ( !*(_QWORD *)(a1 - 96) )
  {
    *(_QWORD *)(a1 - 96) = v3;
    v12 = v3;
    goto LABEL_12;
  }
  v13 = *(_QWORD *)(a1 - 88);
  v12 = v3;
  **(_QWORD **)(a1 - 80) = v13;
  if ( v13 )
    goto LABEL_10;
LABEL_25:
  *(_QWORD *)(a1 - 96) = v12;
LABEL_12:
  v14 = *(_QWORD *)(v12 + 16);
  *(_QWORD *)(a1 - 88) = v14;
  if ( v14 )
    *(_QWORD *)(v14 + 16) = a1 - 88;
  *(_QWORD *)(a1 - 80) = v12 + 16;
  *(_QWORD *)(v12 + 16) = a1 - 96;
  return sub_B4CC70(a1);
}
