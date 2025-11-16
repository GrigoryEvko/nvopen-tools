// Function: sub_22A6120
// Address: 0x22a6120
//
char *__fastcall sub_22A6120(int a1)
{
  char *result; // rax

  switch ( a1 )
  {
    case 0:
    case 19:
      result = "<invalid>";
      break;
    case 1:
      result = "Texture1D";
      break;
    case 2:
      result = "Texture2D";
      break;
    case 3:
      result = "Texture2DMS";
      break;
    case 4:
      result = "Texture3D";
      break;
    case 5:
      result = "TextureCube";
      break;
    case 6:
      result = "Texture1DArray";
      break;
    case 7:
      result = "Texture2DArray";
      break;
    case 8:
      result = "Texture2DMSArray";
      break;
    case 9:
      result = "TextureCubeArray";
      break;
    case 10:
      result = "TypedBuffer";
      break;
    case 11:
      result = "RawBuffer";
      break;
    case 12:
      result = "StructuredBuffer";
      break;
    case 13:
      result = "CBuffer";
      break;
    case 14:
      result = "Sampler";
      break;
    case 15:
      result = "TBuffer";
      break;
    case 16:
      result = "RTAccelerationStructure";
      break;
    case 17:
      result = "FeedbackTexture2D";
      break;
    case 18:
      result = "FeedbackTexture2DArray";
      break;
    default:
      BUG();
  }
  return result;
}
