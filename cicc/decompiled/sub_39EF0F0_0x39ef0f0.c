// Function: sub_39EF0F0
// Address: 0x39ef0f0
//
__int64 __fastcall sub_39EF0F0(__int64 a1, __int64 a2, int a3)
{
  int v5; // esi
  int v7; // eax
  int v8; // edx
  int *v9; // rax
  int v10; // eax
  int v11; // edx
  int *v12; // rax
  int v13; // eax
  int v14; // edx
  int *v15; // rax
  int v16; // eax
  int v17; // edx
  int *v18; // rax
  int v19; // eax
  int v20; // edx
  int *v21; // rax
  int v22; // eax
  int v23; // edx
  int v24; // esi
  int *v25; // rax

  sub_390D5F0(*(_QWORD *)(a1 + 264), a2, 0);
  switch ( a3 )
  {
    case 0:
    case 10:
    case 12:
    case 15:
    case 17:
    case 19:
    case 21:
    case 23:
      return 0;
    case 1:
      v7 = sub_38E2700(a2);
      v8 = 0;
      v5 = v7;
      v9 = (int *)&unk_4534DD0;
      if ( v5 )
      {
        do
        {
          if ( v8 == 2 )
            goto LABEL_3;
          if ( ++v9 == (int *)jpt_39F29C9 )
            break;
          v8 = *v9;
        }
        while ( v5 != *v9 );
      }
      v5 = 2;
      goto LABEL_3;
    case 2:
      v10 = sub_38E2700(a2);
      v11 = 0;
      v5 = v10;
      v12 = (int *)&unk_4534DD0;
      if ( !v5 )
        goto LABEL_16;
      while ( v11 != 10 )
      {
        if ( jpt_39F29C9 != (_UNKNOWN *__ptr32 *)++v12 )
        {
          v11 = *v12;
          if ( v5 != *v12 )
            continue;
        }
LABEL_16:
        v5 = 10;
        goto LABEL_3;
      }
      goto LABEL_3;
    case 3:
      v13 = sub_38E2700(a2);
      v14 = 0;
      v5 = v13;
      v15 = (int *)&unk_4534DD0;
      if ( !v5 )
        goto LABEL_21;
      while ( v14 != 1 )
      {
        if ( jpt_39F29C9 != (_UNKNOWN *__ptr32 *)++v15 )
        {
          v14 = *v15;
          if ( v5 != *v15 )
            continue;
        }
        goto LABEL_21;
      }
      goto LABEL_3;
    case 4:
      v16 = sub_38E2700(a2);
      v17 = 0;
      v5 = v16;
      v18 = (int *)&unk_4534DD0;
      if ( !v5 )
        goto LABEL_26;
      while ( v17 != 6 )
      {
        if ( ++v18 != (int *)jpt_39F29C9 )
        {
          v17 = *v18;
          if ( v5 != *v18 )
            continue;
        }
LABEL_26:
        v5 = 6;
        goto LABEL_3;
      }
      goto LABEL_3;
    case 5:
      v19 = sub_38E2700(a2);
      v20 = 0;
      v5 = v19;
      v21 = (int *)&unk_4534DD0;
      if ( !v5 )
        goto LABEL_21;
      while ( v20 != 1 )
      {
        if ( ++v21 != (int *)jpt_39F29C9 )
        {
          v20 = *v21;
          if ( v5 != *v21 )
            continue;
        }
LABEL_21:
        v5 = 1;
        break;
      }
LABEL_3:
      sub_38E28A0(a2, v5);
      return 1;
    case 6:
      v5 = sub_38E2700(a2);
      goto LABEL_3;
    case 7:
      v22 = sub_38E2700(a2);
      v23 = 0;
      v24 = v22;
      v25 = (int *)&unk_4534DD0;
      if ( !v24 )
        goto LABEL_40;
      break;
    case 8:
      sub_38E2920(a2, 1u);
      *(_BYTE *)(a2 + 8) |= 0x10u;
      return 1;
    case 9:
      sub_38E2720(a2, 2);
      return 1;
    case 11:
      sub_38E2720(a2, 1);
      return 1;
    case 13:
      sub_38E2920(a2, 0);
      *(_BYTE *)(a2 + 8) &= ~0x10u;
      return 1;
    case 18:
      sub_38E2720(a2, 3);
      return 1;
    case 20:
    case 22:
      sub_38E2920(a2, 2u);
      *(_BYTE *)(a2 + 8) |= 0x10u;
      return 1;
    default:
      return 1;
  }
  while ( v23 != 1 )
  {
    if ( ++v25 != (int *)jpt_39F29C9 )
    {
      v23 = *v25;
      if ( v24 != *v25 )
        continue;
    }
LABEL_40:
    v24 = 1;
    break;
  }
  sub_38E28A0(a2, v24);
  sub_38E2920(a2, 0xAu);
  *(_BYTE *)(a2 + 8) |= 0x10u;
  return 1;
}
